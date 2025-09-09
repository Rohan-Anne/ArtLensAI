# backend/scripts/build_index.py
import os, json, faiss, numpy as np
from PIL import Image, ImageEnhance
import torch, open_clip, random

GALLERY_DIR = "backend/models/art_gallery"
OUT_INDEX = "backend/models/index.faiss"
OUT_MAPPING = "backend/models/mapping.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

# Light, fast augmentations to cover phone-screen conditions
def aug_variants(img: Image.Image):
    imgs = [img]
    # brightness
    imgs.append(ImageEnhance.Brightness(img).enhance(0.85))
    imgs.append(ImageEnhance.Brightness(img).enhance(1.15))
    # contrast
    imgs.append(ImageEnhance.Contrast(img).enhance(0.9))
    # tiny rotations
    imgs.append(img.rotate(2, resample=Image.BICUBIC, expand=True))
    imgs.append(img.rotate(-2, resample=Image.BICUBIC, expand=True))
    # mild crop & resize (simulate framing)
    w, h = img.size
    cw, ch = int(w * 0.92), int(h * 0.92)
    x1 = (w - cw) // 2
    y1 = (h - ch) // 2
    imgs.append(img.crop((x1, y1, x1 + cw, y1 + ch)))
    return imgs

def main():
    gallery_path = os.path.join(GALLERY_DIR, "gallery.json")
    if not os.path.exists(gallery_path):
        raise FileNotFoundError(f"Missing {gallery_path}")

    with open(gallery_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )

    embs = []
    mapping = {}
    kept = 0

    for item in items:
        img_path = os.path.join(GALLERY_DIR, item["file"])
        if not os.path.exists(img_path):
            print(f"[WARN] Skipping missing image: {img_path}")
            continue

        base = Image.open(img_path).convert("RGB")
        variants = aug_variants(base)

        for v in variants:
            v = v.convert("RGB")
            with torch.no_grad():
                tens = preprocess(v).unsqueeze(0).to(DEVICE)
                feat = model.encode_image(tens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                vec = feat.detach().cpu().numpy().astype("float32")[0]
            embs.append(vec)
            mapping[str(kept)] = {
                "id": item["id"],                 # group by this later
                "title": item.get("title"),
                "artist": item.get("artist"),
                "year": item.get("year"),
                "thumbnail": f"/static/{item['file']}"
            }
            kept += 1

    if kept == 0:
        raise RuntimeError("No vectors written. Check gallery.json and images.")

    mat = np.stack(embs, axis=0)  # (N, d)
    d = mat.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine on L2-normalized vectors
    index.add(mat)

    faiss.write_index(index, OUT_INDEX)
    with open(OUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"[OK] Built index with {kept} vectors for {len(items)} artworks.")
    print(f"[OK] Wrote {OUT_INDEX} and {OUT_MAPPING}")

if __name__ == "__main__":
    main()


