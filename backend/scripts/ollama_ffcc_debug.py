# backend/scripts/ollama_ffcc_debug.py
import os, time, json, requests, argparse

BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))

sys_prompt = (
    "You are an AP Art History–style art historian. Using broad art-historical knowledge, "
    "write a concise but substantial overview of the artwork using the FFCC structure:\n"
    "Form — materials/technique, style/movement, composition, formal elements (line, color, light, space).\n"
    "Function — intended purpose, patronage/audience, original setting or use.\n"
    "Content — subject matter, iconography/symbols, narrative or meaning.\n"
    "Context — historical period, culture, artist circumstances/innovations, reception/influence.\n\n"
    "Requirements:\n"
    "• 150–220 words total.\n"
    "• Use widely known facts; avoid niche specifics unless famous.\n"
    "• If uncertain, use cautious phrasing ('often interpreted as…').\n"
    "• Do not include disclaimers or mention being an AI.\n"
    "• After the four sections, add exactly two bullet points starting with '• ' labeled as Fast facts."
)

def call_chat(user_prompt, num_predict=400, num_ctx=1024, keep_alive="2h", temperature=0.5):
    url = f"{BASE}/api/chat"
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": keep_alive,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": temperature,
        }
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    ms = (time.time() - t0) * 1000
    r.raise_for_status()
    data = r.json() or {}
    text = ((data.get("message") or {}).get("content") or "").strip()
    return text, ms

def call_generate(user_prompt, num_predict=320, num_ctx=1024, keep_alive="2h", temperature=0.5):
    url = f"{BASE}/api/generate"
    payload = {
        "model": MODEL,
        "stream": False,
        "keep_alive": keep_alive,
        "prompt": user_prompt,
        "system": sys_prompt,
        "options": {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": temperature,
        }
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    ms = (time.time() - t0) * 1000
    r.raise_for_status()
    data = r.json() or {}
    text = (data.get("response") or "").strip()
    return text, ms

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="The Gulf Stream")
    ap.add_argument("--artist", default="Winslow Homer")
    ap.add_argument("--year", default="1899")
    ap.add_argument("--predict", type=int, default=320)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--keep", default="2h")
    ap.add_argument("--endpoint", choices=["chat","gen","both"], default="both")
    args = ap.parse_args()

    user_prompt = (
        f"Title: {args.title}\n"
        f"Artist: {args.artist}\n"
        f"Year: {args.year}\n\n"
        "Write the response with headers exactly:\n"
        "Form:\nFunction:\nContent:\nContext:\n\nFast facts:\n• ...\n• ..."
    )

    print(json.dumps({"base": BASE, "model": MODEL, "timeout_s": TIMEOUT}, indent=2))
    if args.endpoint in ("chat","both"):
        try:
            text, ms = call_chat(user_prompt, num_predict=args.predict, num_ctx=args.ctx, keep_alive=args.keep)
            print(json.dumps({"endpoint": "/api/chat", "ok": True, "ms": ms, "len": len(text)}, indent=2))
            print("\n--- /api/chat FULL OUTPUT ---\n")
            print(text)
            print("\n--- end ---\n")
        except Exception as e:
            print(json.dumps({"endpoint": "/api/chat", "ok": False, "error": str(e)}, indent=2))

    if args.endpoint in ("gen","both"):
        try:
            text, ms = call_generate(user_prompt, num_predict=args.predict, num_ctx=args.ctx, keep_alive=args.keep)
            print(json.dumps({"endpoint": "/api/generate", "ok": True, "ms": ms, "len": len(text)}, indent=2))
            print("\n--- /api/generate FULL OUTPUT ---\n")
            print(text)
            print("\n--- end ---\n")
        except Exception as e:
            print(json.dumps({"endpoint": "/api/generate", "ok": False, "error": str(e)}, indent=2))

if __name__ == "__main__":
    main()
