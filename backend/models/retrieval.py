# backend/models/retrieval.py
import requests, cv2, numpy as np
from PIL import Image, ImageOps
import torch, open_clip
from backend.db_mongo import search_vector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
_model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)

def _embed_pil(img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB")
    with torch.no_grad():
        t = _preprocess(img).unsqueeze(0).to(DEVICE)
        f = _model.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.detach().cpu().numpy()[0].astype("float32")

# ----- geometry + ROI -----
def _order_corners(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

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

def _pad_box(x1,y1,x2,y2,w,h,p=0.12):
    bw, bh = x2-x1, y2-y1
    return (max(0,int(x1-p*bw)), max(0,int(y1-p*bh)),
            min(w-1,int(x2+p*bw)), min(h-1,int(y2+p*bh)))

def find_artwork_roi(pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, 1); edges = cv2.erode(edges, None, 1)
    cnts,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    quad, box = None, None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            pts = approx.reshape(4,2).astype("float32")
            x1,y1 = np.clip(np.min(pts,0), 0, [w-1,h-1])
            x2,y2 = np.clip(np.max(pts,0), 0, [w-1,h-1])
            if (x2-x1)*(y2-y1) < 0.05*w*h: continue
            quad, box = pts, (int(x1),int(y1),int(x2),int(y2)); break

    if quad is not None:
        crop = _perspective_warp(img, quad, 512); x1,y1,x2,y2 = box
    else:
        crop = _center_crop(pil_img, 0.8)
        side = int(0.8*min(h,w)); cx,cy = w//2, h//2
        x1,y1 = max(0,cx-side//2), max(0,cy-side//2); x2,y2 = min(w-1,x1+side), min(h-1,y1+side)

    nbbox = {"x": x1/w, "y": y1/h, "w": (x2-x1)/w, "h": (y2-y1)/h}
    return crop, nbbox, (x1,y1,x2,y2), img

# ----- ORB verify (URL-aware) -----
_orb = cv2.ORB_create(nfeatures=800)

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
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    if len(good) < 10: return 0

    src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kc[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None: return 0
    return int(mask.sum())

# ----- Public API -----
def warmup_index():
    return True  # keep app.py happy

def analyze_image(pil_img: Image.Image, topk: int = 3):
    crop, nbbox, (x1,y1,x2,y2), rgb = find_artwork_roi(pil_img)
    h,w = rgb.shape[:2]
    tight = crop
    px1,py1,px2,py2 = _pad_box(x1,y1,x2,y2,w,h,0.12)
    padded = Image.fromarray(rgb[py1:py2, px1:px2]).convert("RGB").resize((512,512), Image.LANCZOS)
    fallback = _center_crop(pil_img, 0.75).resize((512,512), Image.LANCZOS)

    vecs = [_embed_pil(tight), _embed_pil(padded), _embed_pil(fallback)]

    # Search each crop, fuse by artwork_id using max 'score'
    fused = {}
    metas = {}
    for v in vecs:
        rows = search_vector(v, limit=25, num_candidates=400)
        for r in rows:
            score = float(r.get("score", 0.0))  # higher is better
            aid = r["artwork_id"]
            if score > fused.get(aid, -1e9):
                fused[aid] = score
                metas[aid] = r

    if not fused:
        return {"bbox": nbbox, "candidates": []}

    # ORB re-rank top few
    top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:8]
    rescored = []
    for aid, base_score in top:
        m = metas[aid]
        inl = _orb_inliers(tight, m.get("image_url","") or "")
        bonus = min(inl / 60.0, 0.25)
        rescored.append((float(base_score)+bonus, m))
    rescored.sort(key=lambda t: t[0], reverse=True)

    out = []
    for sc, m in rescored[:topk]:
        out.append({
            "artwork_id": m["artwork_id"],
            "title": m.get("title"),
            "artist": m.get("artist"),
            "year": m.get("year"),
            "thumbnail": m.get("image_url"),
            "score": float(sc),
        })
    return {"bbox": nbbox, "candidates": out}





