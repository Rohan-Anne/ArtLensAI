# backend/scripts/build_index_from_apis.py
import os, json, io, time, random
import requests
from PIL import Image, ImageEnhance
import numpy as np
import faiss
import torch, open_clip

OUT_INDEX = "backend/models/index.faiss"
OUT_MAPPING = "backend/models/mapping.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
_model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)

def _embed(pil: Image.Image) -> np.ndarray:
    with torch.no_grad():
        t = _preprocess(pil.convert("RGB")).unsqueeze(0).to(DEVICE)
        f = _model.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.detach().cpu().numpy().astype("float32")[0]

def _augs(img: Image.Image):
    return [
        img,
        ImageEnhance.Brightness(img).enhance(0.9),
        ImageEnhance.Brightness(img).enhance(1.1),
        ImageEnhance.Contrast(img).enhance(0.92),
        img.rotate(2, resample=Image.BICUBIC, expand=True),
        img.rotate(-2, resample=Image.BICUBIC, expand=True),
    ]

# ---------------- The Met ----------------
# API docs show /objects returns all IDs; /objects/{id} has isPublicDomain + image URLs. (No API key.) :contentReference[oaicite:2]{index=2}
def fetch_met(max_items=1000, seed=42):
    base = "https://collectionapi.metmuseum.org/public/collection/v1"
    ids = requests.get(f"{base}/objects", timeout=20).json().get("objectIDs", [])
    if not ids: return []
    random.Random(seed).shuffle(ids)
    out = []
    for oid in ids:
        try:
            d = requests.get(f"{base}/objects/{oid}", timeout=20).json()
        except Exception:
            continue
        if not d.get("isPublicDomain"): continue
        url = d.get("primaryImage") or d.get("primaryImageSmall")
        if not url: continue
        out.append({
            "id": f"met_{oid}",
            "title": d.get("title"),
            "artist": d.get("artistDisplayName"),
            "year": d.get("objectDate"),
            "image_url": url,
            "source": "met",
        })
        if len(out) >= max_items: break
    return out

# ---------------- AIC ----------------
# AIC docs show how to form IIIF URLs: {iiif_url}/{image_id}/full/843,/0/default.jpg and recommend 843 width. :contentReference[oaicite:3]{index=3}
def fetch_aic(max_items=1000, page_limit=200):
    out = []
    page = 1
    while len(out) < max_items and page <= page_limit:
        try:
            r = requests.get(
                "https://api.artic.edu/api/v1/artworks/search",
                params={
                    "q": "",  # blank query → everything
                    "limit": 100,
                    "page": page,
                    "fields": "id,title,artist_title,date_display,image_id,is_public_domain",
                    "query[term][is_public_domain]": "true",
                    "query[exists][field]": "image_id",
                },
                timeout=20
            ).json()
        except Exception:
            page += 1
            continue

        data = r.get("data", [])
        cfg = r.get("config", {})  # may be present on other endpoints; fall back to default
        iiif_url = cfg.get("iiif_url", "https://www.artic.edu/iiif/2")
        for it in data:
            img_id = it.get("image_id")
            if not img_id or not it.get("is_public_domain"): 
                continue
            url = f"{iiif_url}/{img_id}/full/843,/0/default.jpg"
            out.append({
                "id": f"aic_{it['id']}",
                "title": it.get("title"),
                "artist": it.get("artist_title"),
                "year": it.get("date_display"),
                "image_url": url,
                "source": "aic",
            })
            if len(out) >= max_items:
                break
        page += 1
    return out

def _download_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--met", type=int, default=800, help="How many Met artworks to ingest")
    ap.add_argument("--aic", type=int, default=800, help="How many AIC artworks to ingest")
    args = ap.parse_args()

    items = []
    print("[*] Fetching Met…")
    items += fetch_met(args.met)
    print(f"[OK] Met: {len(items)} total so far")
    print("[*] Fetching AIC…")
    items += fetch_aic(args.aic)
    print(f"[OK] Combined items: {len(items)}")

    if not items:
        raise SystemExit("No items fetched – check your network or lower limits.")

    embs = []
    mapping = {}
    idx = 0
    for it in items:
        img = _download_image(it["image_url"])
        if img is None:
            continue
        for v in _augs(img):
            vec = _embed(v)
            embs.append(vec)
            mapping[str(idx)] = {
                "id": it["id"],
                "title": it.get("title"),
                "artist": it.get("artist"),
                "year": it.get("year"),
                "thumbnail": it["image_url"],  # remote URL
                "source": it.get("source"),
            }
            idx += 1

    if not embs:
        raise SystemExit("No vectors created.")

    mat = np.stack(embs, axis=0)  # (N, 512)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)
    faiss.write_index(index, OUT_INDEX)
    with open(OUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {OUT_INDEX} and {OUT_MAPPING} with {mat.shape[0]} vectors from {len(items)} artworks.")

if __name__ == "__main__":
    main()
