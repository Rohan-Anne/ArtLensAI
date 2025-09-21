# backend/models/retrieval.py
import requests, cv2, numpy as np
from PIL import Image, ImageOps
import torch, open_clip
from backend.db_mongo import search_vector

# ----- CLIP setup -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
_model, _, _preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
)
_model.eval()

@torch.no_grad()
def _embed_pil(img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB")
    t = _preprocess(img).unsqueeze(0).to(DEVICE)
    f = _model.encode_image(t)
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
    return f.detach().cpu().numpy()[0].astype("float32")

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    # a and b should already be L2-normalized; guard anyway
    na = a / (np.linalg.norm(a) + 1e-8)
    nb = b / (np.linalg.norm(b) + 1e-8)
    return float(np.clip((na * nb).sum(), -1.0, 1.0))

# ----- geometry + ROI -----
def _order_corners(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def _quad_is_reasonable(pts, w, h):
    x1,y1 = np.clip(np.min(pts,0), 0, [w-1,h-1])
    x2,y2 = np.clip(np.max(pts,0), 0, [w-1,h-1])
    area = (x2-x1) * (y2-y1)
    if area < 0.05 * w * h: 
        return False
    ar = (x2-x1) / max(1e-6, (y2-y1))
    return 0.5 <= ar <= 2.0

def _perspective_warp(rgb, quad, out=512):
    rect = _order_corners(quad); (tl,tr,br,bl) = rect
    wA = np.linalg.norm(br-bl); wB = np.linalg.norm(tr-tl)
    hA = np.linalg.norm(tr-br); hB = np.linalg.norm(tl-bl)
    W = max(int(wA), int(wB), 32); H = max(int(hA), int(hB), 32)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(rgb, M, (W, H))
    warped = cv2.resize(warped, (out, out), interpolation=cv2.INTER_AREA)
    return Image.fromarray(warped)

def _center_crop(pil, pct=0.8):
    w,h = pil.size; side = int(min(w,h)*pct)
    x1 = max(0,(w-side)//2); y1 = max(0,(h-side)//2)
    return pil.crop((x1,y1,x1+side,y1+side))

def _pad_box(x1,y1,x2,y2,w,h,p=0.14):
    bw, bh = x2-x1, y2-y1
    return (max(0,int(x1-p*bw)), max(0,int(y1-p*bh)),
            min(w,int(x2+p*bw)), min(h,int(y2+p*bh)))

def find_artwork_roi(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    h,w = img.shape[:2]

    # CLAHE for steadier edges under lighting changes
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    gray = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, 1); edges = cv2.erode(edges, None, 1)
    cnts,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:12]

    quad, box = None, None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            pts = approx.reshape(4,2).astype("float32")
            if not _quad_is_reasonable(pts, w, h): 
                continue
            x1,y1 = np.clip(np.min(pts,0), 0, [w-1,h-1])
            x2,y2 = np.clip(np.max(pts,0), 0, [w-1,h-1])
            quad, box = pts, (int(x1),int(y1),int(x2),int(y2))
            break

    if quad is not None:
        crop = _perspective_warp(img, quad, 512); x1,y1,x2,y2 = box
    else:
        crop = _center_crop(pil_img, 0.82)
        side = int(0.82*min(h,w)); cx,cy = w//2, h//2
        x1,y1 = max(0,cx-side//2), max(0,cy-side//2)
        x2,y2 = min(w-1,x1+side), min(h-1,y1+side)

    nbbox = {"x": x1/w, "y": y1/h, "w": (x2-x1)/w, "h": (y2-y1)/h}
    return crop, nbbox, (x1,y1,x2,y2), img

# ----- ORB verify (URL-aware) -----
_orb = cv2.ORB_create(nfeatures=1200)

def _imread_url(url):
    try:
        r = requests.get(url, timeout=7); r.raise_for_status()
        arr = np.frombuffer(r.content, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None: return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def _orb_inliers(query_pil: Image.Image, url: str) -> int:
    q = np.array(query_pil.convert("RGB"))
    c = _imread_url(url)
    if c is None: return 0

    def prep(im):
        h,w = im.shape[:2]; s = 640/max(h,w)
        if s < 1: im = cv2.resize(im, (int(w*s), int(h*s)), cv2.INTER_AREA)
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    qg, cg = prep(q), prep(c)
    kq, dq = _orb.detectAndCompute(qg, None)
    kc, dc = _orb.detectAndCompute(cg, None)
    if dq is None or dc is None or len(kq)<8 or len(kc)<8: return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(dq, dc, k=2)
    good = [m for m,n in matches if n is not None and m.distance < 0.8*n.distance]
    if len(good) < 12: return 0

    src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kc[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None: return 0
    return int(mask.sum())

# ----- score helpers (bounded to [0,1]) -----
def _extract_backend_score(row: dict) -> float:
    # Try typical fields; DO NOT bound here. We'll normalize per-batch later.
    if "score" in row and row["score"] is not None:
        return float(row["score"])
    if "distance" in row and row["distance"] is not None:
        # lower is better; flip to similarity-ish but still raw
        return 1.0 - float(row["distance"])
    if "metric" in row and row["metric"] is not None:
        try: return 1.0 - float(row["metric"])
        except Exception: pass
    return float("-inf")

def _minmax_norm(vals):
    v = np.array(vals, dtype=np.float32)
    if len(v)==0: return v
    v[~np.isfinite(v)] = np.nan
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.zeros_like(v)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi - lo < 1e-9:
        return np.ones_like(v) * 0.5
    out = (v - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return out

def _clip_verify_similarity(query_emb: np.ndarray, url: str) -> float:
    """Return CLIP image-image cosine in [0,1] for a candidate thumbnail URL."""
    arr = _imread_url(url)
    if arr is None: 
        return 0.0
    sim = _cos(query_emb, _embed_pil(Image.fromarray(arr)))
    # map cosine [-1,1] -> [0,1]
    return 0.5 * (sim + 1.0)

# ----- Public API -----
def warmup_index():
    return True  # keep app.py happy

def analyze_image(pil_img: Image.Image, topk: int = 3):
    crop, nbbox, (x1,y1,x2,y2), rgb = find_artwork_roi(pil_img)
    h,w = rgb.shape[:2]

    tight = crop
    px1,py1,px2,py2 = _pad_box(x1,y1,x2,y2,w,h,0.14)
    padded_arr = rgb[py1:py2, px1:px2]
    if padded_arr.size == 0:
        padded_arr = rgb
    padded = Image.fromarray(padded_arr).convert("RGB").resize((512,512), Image.LANCZOS)
    fallback = _center_crop(pil_img, 0.78).resize((512,512), Image.LANCZOS)

    crops = [tight, padded, fallback]
    vecs = [_embed_pil(c) for c in crops]

    # 1) Retrieve with each crop
    fused = {}
    metas = {}
    hits = {}
    backend_scores_all = {}  # aid -> list of raw backend scores (for later min-max)
    for v in vecs:
        rows = search_vector(v, limit=30, num_candidates=600)
        # Gather raw scores for local normalization
        raw_scores = [ _extract_backend_score(r) for r in rows ]
        norm = _minmax_norm(raw_scores)
        for r, s_norm in zip(rows, norm):
            aid = r.get("artwork_id")
            if not aid:
                continue
            # Keep best normalized backend score observed across crops
            prev = fused.get(aid, -1.0)
            if s_norm > prev:
                fused[aid] = float(s_norm)     # ∈ [0,1]
                metas[aid] = r
            hits[aid] = hits.get(aid, 0) + 1
            backend_scores_all.setdefault(aid, []).append(float(s_norm))

    if not fused:
        return {"bbox": nbbox, "candidates": []}

    # 2) Second-stage verification on top candidates: CLIP image-image + ORB
    # Only verify a small pool for speed.
    pool = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:12]
    q_emb = _embed_pil(tight)

    rescored = []
    for aid, db_norm in pool:
        m = metas[aid]
        url = (m.get("image_url") or "").strip()
        clip_sim = _clip_verify_similarity(q_emb, url) if url else 0.0  # ∈ [0,1]
        inl = _orb_inliers(tight, url) if url else 0
        geo = min(inl / 80.0, 1.0)  # map to [0,1], gentle scale

        agree = min(max(0, hits.get(aid,1)-1) * 0.5, 1.0)  # (#extra crops)*0.5 capped, then
        agree = agree * 0.1  # final weight is small; treat as [0,0.1]

        # Compose a bounded final score in [0,1]
        final = 0.60*clip_sim + 0.25*db_norm + 0.10*geo + 0.05*(agree/0.1)
        final = float(np.clip(final, 0.0, 1.0))

        rescored.append((final, m))

    rescored.sort(key=lambda t: t[0], reverse=True)

    out = []
    for sc, m in rescored[:topk]:
        out.append({
            "artwork_id": m["artwork_id"],
            "title": m.get("title"),
            "artist": m.get("artist"),
            "year": m.get("year"),
            "thumbnail": m.get("image_url"),
            "score": sc,   # guaranteed 0..1
        })
    return {"bbox": nbbox, "candidates": out}
