# backend/app.py
import os, json, re, time, logging, requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from PIL import Image

# Retrieval pipeline
from backend.models.retrieval import analyze_image, warmup_index

# Optional cache (Mongo)
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ---------------- logging ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
LLM_DEBUG = os.getenv("LLM_DEBUG", "0") == "1"

# ---------------- Together config ----------------
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo").strip()
TOGETHER_BASE = os.getenv("TOGETHER_BASE", "https://api.together.xyz").rstrip("/")

MAX_TOKENS = int(os.getenv("TOGETHER_MAX_TOKENS", "1000")) 
TEMPERATURE = float(os.getenv("TOGETHER_TEMPERATURE", "0.5"))

if not TOGETHER_API_KEY:
    logging.warning("[llm] TOGETHER_API_KEY not set; streaming endpoints will fail")

# ---------------- Mongo cache (optional) ----------------
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "artspot")
_client = None
_sum_coll = None
if MONGODB_URI:
    try:
        _client = MongoClient(MONGODB_URI)
        _sum_coll = _client[MONGODB_DB]["summaries"]
        logging.info("[cache] Mongo summaries collection is ready")
    except Exception as e:
        logging.warning(f"[cache] Mongo not available: {e}")
        _client = None
        _sum_coll = None

# ---------------- FFCC detector ----------------
_FFCC_RE = re.compile(r"Form:.*?Function:.*?Content:.*?Context:", re.IGNORECASE | re.DOTALL)
def _is_ffcc(txt: str) -> bool:
    return bool(txt) and bool(_FFCC_RE.search(txt))

def _aid_of(art: dict) -> str:
    return art.get("artwork_id") or art.get("id") or f"{(art.get('title') or '').strip()}|{(art.get('artist') or '').strip()}"

# ---------------- cache helpers (validated) ----------------
def _cache_get_valid(aid: str) -> str | None:
    """Return cached text only if it looks like a real FFCC summary."""
    if _sum_coll is None or not aid:
        return None
    try:
        doc = _sum_coll.find_one({"_id": aid}) or {}
        txt = doc.get("text") or ""
        if _is_ffcc(txt):
            if LLM_DEBUG:
                logging.info(f"[cache] valid hit for {aid!r} (len={len(txt)})")
            return txt
        if txt:
            _sum_coll.delete_one({"_id": aid})  # drop stale/non-FFCC cache
            if LLM_DEBUG:
                logging.info(f"[cache] dropped stale cache for {aid!r}")
        return None
    except PyMongoError as e:
        logging.warning(f"[cache] get error: {e}")
        return None

def _cache_put(aid: str, text: str) -> None:
    if _sum_coll is None or not aid or not _is_ffcc(text):
        return
    try:
        _sum_coll.update_one({"_id": aid}, {"$set": {"text": text}}, upsert=True)
        if LLM_DEBUG:
            logging.info(f"[cache] stored FFCC for {aid!r} (len={len(text)})")
    except PyMongoError as e:
        logging.warning(f"[cache] put error: {e}")

# ---------------- prompt builder ----------------
def _build_messages(art: dict):
    title  = (art.get("title") or "Unknown").strip()
    artist = (art.get("artist") or "Unknown").strip()
    year   = str(art.get("year") or "Unknown").strip()

    system = (
        "You are an AP Art History teacher. Write a compact FFCC overview.\n"
        "Use four labeled sections (Form, Function, Content, Context), then exactly two 'Fast facts' bullets.\n"
        "Prefer widely known facts; when unsure, use cautious phrasing (e.g., 'often interpreted as…').\n"
        "Do not include disclaimers or mention being an AI."
    )

    user = (
        f"Title: {title}\n"
        f"Artist: {artist}\n"
        f"Year: {year}\n\n"
        "Write the response with headers exactly:\n"
        "Form:\n"
        "Function:\n"
        "Content:\n"
        "Context:\n\n"
        "Fast facts:\n"
        "• ...\n"
        "• ...\n"
        "Stop after the second bullet."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ---------------- Together calls ----------------
def _together_nonstream_chat(messages):
    url = f"{TOGETHER_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TOGETHER_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json() or {}
    ch = (data.get("choices") or [{}])[0]
    return (ch.get("message") or {}).get("content") or ""

def _together_stream_chat(messages):
    """
    Yields incremental text chunks (OpenAI-style SSE).
    """
    url = f"{TOGETHER_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": TOGETHER_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data = raw[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    js = json.loads(data)
                    delta = ((js.get("choices") or [{}])[0].get("delta") or {})
                    piece = delta.get("content") or ""
                    if piece:
                        yield piece
                except Exception:
                    continue

# ---------------- Flask helpers (SSE) ----------------
def _sse(event: str, data):
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

# ---------------- Summarizers ----------------
def summarize_nonstream(art: dict) -> str:
    aid = _aid_of(art)
    if aid:
        hit = _cache_get_valid(aid)
        if hit:
            return hit

    msgs = _build_messages(art)
    try:
        text = (_together_nonstream_chat(msgs) or "").strip()
    except Exception as e:
        logging.warning(f"[llm] together nonstream failed: {e}")
        text = ""

    if _is_ffcc(text) and aid:
        _cache_put(aid, text)
    return text if _is_ffcc(text) else _fallback_text(art)

def stream_summary_sse(art: dict):
    aid = _aid_of(art)
    want_nocache = bool((art or {}).get("nocache")) or bool(request.args.get("nocache"))

    def gen():
        yield _sse("meta", {"model": TOGETHER_MODEL})
        if aid and not want_nocache:
            cached = _cache_get_valid(aid)
            if cached:
                yield _sse("token", {"text": cached})
                yield _sse("done", {})
                return

        msgs = _build_messages(art or {})
        try:
            full = []
            for piece in _together_stream_chat(msgs):
                full.append(piece)
                yield _sse("token", {"text": piece})
            txt = "".join(full)
            if aid and _is_ffcc(txt):
                _cache_put(aid, txt)
            yield _sse("done", {})
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    resp = Response(gen(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Connection"] = "keep-alive"
    return resp

def _fallback_text(art: dict) -> str:
    title  = (art.get("title") or "").strip()
    artist = (art.get("artist") or "").strip()
    year   = str(art.get("year") or "").strip()
    base = []
    if title: base.append(f"Title: {title}")
    if artist: base.append(f"Artist: {artist}")
    if year: base.append(f"Year: {year}")
    return " • ".join(base) if base else "No metadata available."

# ---------------- Flask app / routes ----------------
app = Flask(__name__, static_folder="models/art_gallery")
CORS(app)

try:
    warmup_index()
except Exception as e:
    logging.warning(f"[init] retrieval warmup failed: {e}")

@app.route("/static/<path:filename>")
def static_proxy(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    img = Image.open(request.files["file"].stream).convert("RGB")
    return jsonify(analyze_image(img, topk=3))

# --- non-stream (kept for compatibility) ---
@app.post("/api/summary")
def summary_legacy():
    data = request.get_json(silent=True) or {}
    art = data.get("artwork") or {}
    text = summarize_nonstream(art) or ""
    return jsonify({"summary": text})

@app.post("/api/summarize")
def summarize_new():
    art = request.get_json(silent=True) or {}
    text = summarize_nonstream(art) or ""
    return jsonify({"summary": text})

# --- NEW: streaming endpoint (SSE) ---
@app.post("/api/summarize/stream")
def summarize_stream():
    art = request.get_json(silent=True) or {}
    return stream_summary_sse(art)

# --- tiny cache purge (optional) ---
@app.delete("/api/cache/purge")
def purge_one():
    aid = request.args.get("aid", "").strip()
    if not aid or _sum_coll is None:
        return jsonify({"ok": False, "msg": "missing aid or cache disabled"}), 400
    try:
        n = _sum_coll.delete_one({"_id": aid}).deleted_count
        return jsonify({"ok": True, "deleted": n})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# --- health ---
@app.get("/api/llm_health")
def llm_health():
    ok = bool(TOGETHER_API_KEY)
    return jsonify({
        "provider": "together",
        "base": TOGETHER_BASE,
        "model": TOGETHER_MODEL,
        "has_key": ok
    })

if __name__ == "__main__":
    logging.info(f"[init] Together model={TOGETHER_MODEL}")
    app.run(host="0.0.0.0", port=5000, debug=True)
