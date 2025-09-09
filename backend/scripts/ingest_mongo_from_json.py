# backend/scripts/ingest_mongo_from_json.py
"""
Usage:
  docker compose exec backend python backend/scripts/ingest_mongo_from_json.py backend/models/artworks_seed.json

JSON format:
[
  {"id":"great_wave_hokusai","title":"The Great Wave off Kanagawa","artist":"Katsushika Hokusai","year":"c. 1830–1832","image_url":"https://...jpg"},
  ...
]
"""

import sys, io, json, requests
from typing import List, Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
import torch, open_clip
from backend.db_mongo import insert_vectors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "ViT-B-32"
PRETRAINED = "openai"
_model, _, _preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=DEVICE)

def embed(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    with torch.no_grad():
        t = _preprocess(img).unsqueeze(0).to(DEVICE)
        f = _model.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)     # L2 normalize
    return f.detach().cpu().numpy().astype("float32")[0]  # (512,)

def augs(img: Image.Image) -> List[Image.Image]:
    return [
        img,
        ImageEnhance.Brightness(img).enhance(0.9),
        ImageEnhance.Brightness(img).enhance(1.1),
        ImageEnhance.Contrast(img).enhance(0.92),
        img.rotate(2, resample=Image.BICUBIC, expand=True),
        img.rotate(-2, resample=Image.BICUBIC, expand=True),
    ]

def download(url: str) -> Image.Image:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def main(path: str):
    with open(path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    batch: List[Dict[str, Any]] = []
    for it in items:
        try:
            img = download(it["image_url"])
        except Exception as e:
            print(f"[WARN] skip {it.get('id')}: download failed ({e})")
            continue
        for v in augs(img):
            vec = embed(v).tolist()
            batch.append({
                "artwork_id": it["id"],
                "title": it.get("title"),
                "artist": it.get("artist"),
                "year": it.get("year"),
                "image_url": it.get("image_url"),
                "embedding": vec
            })

    n = insert_vectors(batch)
    print(f"[OK] inserted {n} vector docs")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide path to seed JSON.")
        sys.exit(1)
    main(sys.argv[1])
