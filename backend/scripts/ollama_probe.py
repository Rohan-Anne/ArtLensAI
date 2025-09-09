#!/usr/bin/env python3
"""
Ollama probe / smoke test.

- Verifies connectivity to the Ollama service
- Lists models and checks the target model exists
- (Optional) pulls a model via /api/pull
- Runs a tiny prompt through multiple endpoints:
    /api/chat, /api/generate, /v1/chat/completions, /v1/completions
- Measures latency and prints a short preview

Exit code 0 = at least one endpoint returned text successfully.
Exit code 1 = all endpoints failed.

Usage (in Docker):
  docker compose exec backend python backend/scripts/ollama_probe.py \
    --base http://ollama:11434 --model llama3.1:8b-instruct-q4_K_M

From host (if Ollama runs on host):
  python backend/scripts/ollama_probe.py --base http://localhost:11434 --model llama3.1:8b-instruct-q4_K_M
"""

import os
import sys
import time
import json
import argparse
import requests

DEFAULT_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))  # seconds

def hr(n):  # human readable ms
    return f"{n:.1f} ms"

def get_json(url, timeout):
    t0 = time.time()
    try:
        r = requests.get(url, timeout=timeout)
        ms = (time.time() - t0) * 1000
        return r, ms, None
    except Exception as e:
        ms = (time.time() - t0) * 1000
        return None, ms, e

def post_json(url, payload, timeout, stream=False):
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, timeout=timeout, stream=stream)
        ms = (time.time() - t0) * 1000
        return r, ms, None
    except Exception as e:
        ms = (time.time() - t0) * 1000
        return None, ms, e

def check_version(base, timeout):
    r, ms, err = get_json(f"{base}/api/version", timeout)
    if err:
        return {"endpoint": "/api/version", "ok": False, "error": str(err), "ms": ms}
    body = (r.text or "").strip()
    return {"endpoint": "/api/version", "ok": r.ok, "status": r.status_code, "ms": ms, "body": body[:200]}

def list_models(base, timeout):
    r, ms, err = get_json(f"{base}/v1/models", timeout)
    if err:
        return {"endpoint": "/v1/models", "ok": False, "error": str(err), "ms": ms, "models": []}
    models = []
    try:
        data = r.json() or {}
        models = [d.get("id") for d in data.get("data", []) if d.get("id")]
    except Exception:
        pass
    return {"endpoint": "/v1/models", "ok": r.ok, "status": r.status_code, "ms": ms, "models": models}

def maybe_pull_model(base, model, timeout):
    """
    Pull a model via /api/pull (streaming NDJSON). Only used when --pull is passed.
    """
    print(f"[*] Pulling model {model!r} from {base} … (this may take a while)")
    payload = {"name": model}
    r, _, err = post_json(f"{base}/api/pull", payload, timeout=timeout, stream=True)
    if err:
        print(f"[pull] error: {err}")
        return False
    if r.status_code >= 400:
        print(f"[pull] HTTP {r.status_code}: {r.text[:300]}")
        return False
    try:
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
                status = obj.get("status") or obj.get("digest") or obj
                print(f"[pull] {status}")
            except Exception:
                print(f"[pull] {line}")
    finally:
        r.close()
    print("[pull] done.")
    return True

def run_api_chat(base, model, prompt, system, timeout, num_ctx=1024, num_predict=120, temperature=0.5):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    r, ms, err = post_json(f"{base}/api/chat", payload, timeout)
    meta = {"endpoint": "/api/chat", "ms": ms}
    if err:
        return None, {**meta, "error": str(err)}
    if r.status_code in (404, 405):
        return None, {**meta, "status": r.status_code}
    try:
        r.raise_for_status()
        data = r.json() or {}
        text = ((data.get("message") or {}).get("content") or "").strip()
        return text, {**meta, "status": r.status_code}
    except Exception as e:
        return None, {**meta, "status": r.status_code, "error": str(e), "body": r.text[:300]}

def run_api_generate(base, model, prompt, system, timeout, num_ctx=1024, num_predict=120, temperature=0.5):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    r, ms, err = post_json(f"{base}/api/generate", payload, timeout)
    meta = {"endpoint": "/api/generate", "ms": ms}
    if err:
        return None, {**meta, "error": str(err)}
    if r.status_code in (404, 405):
        return None, {**meta, "status": r.status_code}
    try:
        r.raise_for_status()
        data = r.json() or {}
        text = (data.get("response") or "").strip()
        return text, {**meta, "status": r.status_code}
    except Exception as e:
        return None, {**meta, "status": r.status_code, "error": str(e), "body": r.text[:300]}

def run_v1_chat(base, model, prompt, system, timeout, max_tokens=160, temperature=0.5):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r, ms, err = post_json(f"{base}/v1/chat/completions", payload, timeout)
    meta = {"endpoint": "/v1/chat/completions", "ms": ms}
    if err:
        return None, {**meta, "error": str(err)}
    if r.status_code in (404, 405):
        return None, {**meta, "status": r.status_code}
    try:
        r.raise_for_status()
        data = r.json() or {}
        ch = (data.get("choices") or [{}])[0]
        text = ((ch.get("message") or {}).get("content") or "").strip()
        return text, {**meta, "status": r.status_code}
    except Exception as e:
        return None, {**meta, "status": r.status_code, "error": str(e), "body": r.text[:300]}

def run_v1_completions(base, model, prompt, system, timeout, max_tokens=160, temperature=0.5):
    stitched = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}\n\n[ASSISTANT]\n"
    payload = {
        "model": model,
        "prompt": stitched,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r, ms, err = post_json(f"{base}/v1/completions", payload, timeout)
    meta = {"endpoint": "/v1/completions", "ms": ms}
    if err:
        return None, {**meta, "error": str(err)}
    if r.status_code in (404, 405):
        return None, {**meta, "status": r.status_code}
    try:
        r.raise_for_status()
        data = r.json() or {}
        ch = (data.get("choices") or [{}])[0]
        text = (ch.get("text") or "").strip()
        return text, {**meta, "status": r.status_code}
    except Exception as e:
        return None, {**meta, "status": r.status_code, "error": str(e), "body": r.text[:300]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE, help="Ollama base URL (http://ollama:11434 or http://localhost:11434)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model id (must exist in Ollama `ollama list`)")
    ap.add_argument("--prompt", default="Reply with the single word: OK.", help="User prompt")
    ap.add_argument("--system", default="You are terse.", help="System prompt")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--pull", action="store_true", help="Attempt to pull the model via /api/pull before testing")
    ap.add_argument("--mode", choices=["auto","chat","generate","v1chat","v1comp"], default="auto",
                    help="Which endpoint to test (auto = try all until one works)")
    ap.add_argument("--num-predict", type=int, default=120)
    ap.add_argument("--num-ctx", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    base = args.base.rstrip("/")
    model = args.model

    print(json.dumps({"base": base, "model": model, "timeout_s": args.timeout}, indent=2))

    # 1) version
    ver = check_version(base, args.timeout)
    print(json.dumps(ver, indent=2))

    # 2) models
    mods = list_models(base, args.timeout)
    print(json.dumps(mods, indent=2))
    available = model in (mods.get("models") or [])

    # 3) pull if needed/asked
    if args.pull and not available:
        ok = maybe_pull_model(base, model, timeout=max(args.timeout, 600))
        if not ok:
            print(json.dumps({"pull": "failed"}, indent=2))
        # refresh list
        mods = list_models(base, args.timeout)
        print(json.dumps(mods, indent=2))
        available = model in (mods.get("models") or [])

    if not available:
        print(json.dumps({"error": f"model '{model}' not found in Ollama"}, indent=2))
        # Continue anyway—some older versions might still accept it by implicit pull.

    # 4) run test(s)
    runners = []
    if args.mode == "chat":
        runners = [run_api_chat]
    elif args.mode == "generate":
        runners = [run_api_generate]
    elif args.mode == "v1chat":
        runners = [run_v1_chat]
    elif args.mode == "v1comp":
        runners = [run_v1_completions]
    else:
        runners = [run_api_chat, run_api_generate, run_v1_chat, run_v1_completions]

    success = False
    for fn in runners:
        try:
            if fn in (run_api_chat, run_api_generate):
                text, meta = fn(base, model, args.prompt, args.system, args.timeout,
                                num_ctx=args.num_ctx, num_predict=args.num_predict, temperature=args.temperature)
            else:
                text, meta = fn(base, model, args.prompt, args.system, args.timeout,
                                max_tokens=args.num_predict, temperature=args.temperature)
            out = {
                "endpoint": meta.get("endpoint"),
                "status": meta.get("status"),
                "ms": meta.get("ms"),
            }
            if text:
                success = True
                out["ok"] = True
                out["preview"] = text
            else:
                out["ok"] = False
                if "error" in meta:
                    out["error"] = meta["error"]
                if "body" in meta:
                    out["bodyPreview"] = meta["body"]
            print(json.dumps(out, indent=2))
            if success and args.mode == "auto":
                break
        except Exception as e:
            print(json.dumps({"endpoint": fn.__name__, "ok": False, "error": str(e)}, indent=2))

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

