# backend/scripts/make_seed_famous.py
"""
Build a seed JSON of famous, public-domain artworks using:
  • The Met Collection API
  • Art Institute of Chicago (AIC) API (IIIF)

Strategy:
  1) Curated title+artist pairs (strict).
  2) Fill remainder by artist (several works each).

Usage:
  docker compose exec backend \
    python backend/scripts/make_seed_famous.py \
      --out backend/models/artworks_seed_120.json \
      --target 120 \
      --per-artist 6
"""

from __future__ import annotations
import os, json, time, argparse, requests, re
from typing import Dict, List, Optional

MET_BASE   = "https://collectionapi.metmuseum.org/public/collection/v1"
AIC_SEARCH = "https://api.artic.edu/api/v1/artworks/search"

# ------------------------- curated iconic pairs (strict) -------------------------
FAMOUS_TITLES: List[Dict[str, str]] = [
  {"title": "The Great Wave off Kanagawa", "artist": "Hokusai"},
  {"title": "Aristotle with a Bust of Homer", "artist": "Rembrandt"},
  {"title": "The Harvesters", "artist": "Pieter Bruegel"},
  {"title": "Washington Crossing the Delaware", "artist": "Emanuel Leutze"},
  {"title": "Bridge over a Pond of Water Lilies", "artist": "Claude Monet"},
  {"title": "Water Lilies", "artist": "Claude Monet"},
  {"title": "Haystacks", "artist": "Claude Monet"},
  {"title": "Rouen Cathedral", "artist": "Claude Monet"},
  {"title": "Madame X", "artist": "John Singer Sargent"},
  {"title": "The Horse Fair", "artist": "Rosa Bonheur"},
  {"title": "The Gulf Stream", "artist": "Winslow Homer"},
  {"title": "The Veteran in a New Field", "artist": "Winslow Homer"},
  {"title": "The Musicians", "artist": "Caravaggio"},
  {"title": "The Denial of Saint Peter", "artist": "Caravaggio"},
  {"title": "Self-Portrait with a Straw Hat", "artist": "Vincent van Gogh"},
  {"title": "Wheat Field with Cypresses", "artist": "Vincent van Gogh"},
  {"title": "The Bedroom", "artist": "Vincent van Gogh"},
  {"title": "Two Sisters (On the Terrace)", "artist": "Pierre-Auguste Renoir"},
  {"title": "Paris Street; Rainy Day", "artist": "Gustave Caillebotte"},
  {"title": "At the Moulin Rouge", "artist": "Henri de Toulouse-Lautrec"},
  {"title": "Ballet Rehearsal", "artist": "Edgar Degas"},
  {"title": "The Banjo Lesson", "artist": "Henry Ossawa Tanner"},
  {"title": "The Oxbow", "artist": "Thomas Cole"},
  {"title": "The Death of Socrates", "artist": "Jacques-Louis David"},
  {"title": "The Thinker", "artist": "Auguste Rodin"},
  {"title": "The Milkmaid", "artist": "Johannes Vermeer"},
  {"title": "Woman with a Water Jug", "artist": "Johannes Vermeer"},
  {"title": "The Night Café", "artist": "Vincent van Gogh"},
  {"title": "The Sower", "artist": "Vincent van Gogh"},
  {"title": "Woman with a Parasol", "artist": "Claude Monet"},
  {"title": "Nocturne in Black and Gold", "artist": "James McNeill Whistler"},
  {"title": "The Fighting Temeraire", "artist": "J. M. W. Turner"},
  {"title": "The Hay Wain", "artist": "John Constable"},
  {"title": "The Arnolfini Portrait", "artist": "Jan van Eyck"},
  {"title": "The Swing", "artist": "Jean-Honoré Fragonard"},
  {"title": "View of Toledo", "artist": "El Greco"},
  {"title": "Judith and Holofernes", "artist": "Artemisia Gentileschi"},
  {"title": "Juan de Pareja", "artist": "Diego Velázquez"},
  {"title": "Liberty Leading the People", "artist": "Eugène Delacroix"},
  {"title": "Olympia", "artist": "Édouard Manet"},
  {"title": "Luncheon on the Grass", "artist": "Édouard Manet"},
  {"title": "The Card Players", "artist": "Paul Cézanne"},
  {"title": "Still Life with Apples", "artist": "Paul Cézanne"},
  {"title": "The Child's Bath", "artist": "Mary Cassatt"},
  {"title": "A Sunday Afternoon on the Island of La Grande Jatte", "artist": "Georges Seurat"},
  {"title": "Stacks of Wheat (End of Summer)", "artist": "Claude Monet"},
  {"title": "Young Woman with a Water Pitcher", "artist": "Johannes Vermeer"},
  {"title": "The Sleepers", "artist": "Gustave Courbet"},
  {"title": "The Third-Class Carriage", "artist": "Honoré Daumier"},
  {"title": "Portrait of Madame Cézanne", "artist": "Paul Cézanne"},
]

# ------------------------- famous artists (broad fetch) -------------------------
FAMOUS_ARTISTS: List[str] = [
  # Northern Renaissance & Baroque
  "Jan van Eyck", "Pieter Bruegel", "Rembrandt", "Johannes Vermeer", "Peter Paul Rubens",
  "Anthony van Dyck", "Albrecht Dürer", "Lucas Cranach", "Frans Hals",
  # Italian Renaissance / Baroque
  "Sandro Botticelli", "Leonardo da Vinci", "Raphael", "Titian", "Tintoretto",
  "Caravaggio", "Artemisia Gentileschi", "Giovanni Bellini",
  # Spanish
  "Diego Velázquez", "Francisco de Goya", "El Greco", "Murillo", "Zurbarán",
  # French 17–19c
  "Nicolas Poussin", "Jean-Honoré Fragonard", "Jean-Antoine Watteau", "François Boucher",
  "Jacques-Louis David", "Jean-Auguste-Dominique Ingres", "Eugène Delacroix",
  "Gustave Courbet", "Jean-Baptiste-Camille Corot",
  # British
  "Thomas Gainsborough", "Joshua Reynolds", "J. M. W. Turner", "John Constable", "William Hogarth",
  # American 19c
  "Thomas Cole", "Frederic Edwin Church", "Winslow Homer", "Thomas Eakins", "James McNeill Whistler",
  # Impressionists/Post-Impressionists
  "Édouard Manet", "Claude Monet", "Pierre-Auguste Renoir", "Edgar Degas", "Berthe Morisot",
  "Gustave Caillebotte", "Camille Pissarro", "Alfred Sisley",
  "Paul Cézanne", "Georges Seurat", "Paul Signac",
  "Vincent van Gogh", "Paul Gauguin", "Henri de Toulouse-Lautrec", "Mary Cassatt",
  # Others 19c
  "Rosa Bonheur", "John Singer Sargent", "James Tissot",
  # Japanese ukiyo-e
  "Katsushika Hokusai", "Utagawa Hiroshige", "Kitagawa Utamaro",
  # Sculpture
  "Auguste Rodin", "Antonio Canova",
]

# ------------------------------ helpers ---------------------------------
def _infix(a: Optional[str], b: Optional[str]) -> bool:
  return bool(a) and bool(b) and a.lower() in b.lower()

def met_get_title(title: str, artist: Optional[str] = None, timeout: int = 25):
  try:
    sr = requests.get(f"{MET_BASE}/search", params={"q": title, "hasImages": "true"}, timeout=timeout).json()
    ids = sr.get("objectIDs") or []
    for oid in ids[:80]:
      d = requests.get(f"{MET_BASE}/objects/{oid}", timeout=timeout).json()
      if not d.get("isPublicDomain"): continue
      img = d.get("primaryImage") or d.get("primaryImageSmall")
      if not img: continue
      if artist and not _infix(artist, d.get("artistDisplayName", "")): continue
      return {
        "id": f"met_{oid}",
        "title": d.get("title"),
        "artist": d.get("artistDisplayName"),
        "year": d.get("objectDate"),
        "image_url": img,
        "source": "met",
      }
  except Exception:
    pass
  return None

def met_get_by_artist(artist: str, want: int = 5, timeout: int = 25) -> List[Dict]:
  out: List[Dict] = []
  try:
    sr = requests.get(
      f"{MET_BASE}/search",
      params={"q": artist, "hasImages": "true", "artistOrCulture": "true"},
      timeout=timeout
    ).json()
    ids = sr.get("objectIDs") or []
    for oid in ids[:300]:
      d = requests.get(f"{MET_BASE}/objects/{oid}", timeout=timeout).json()
      if not d.get("isPublicDomain"): continue
      img = d.get("primaryImage") or d.get("primaryImageSmall")
      if not img: continue
      if not _infix(artist, d.get("artistDisplayName", "")): continue
      out.append({
        "id": f"met_{oid}",
        "title": d.get("title"),
        "artist": d.get("artistDisplayName"),
        "year": d.get("objectDate"),
        "image_url": img,
        "source": "met",
      })
      if len(out) >= want: break
  except Exception:
    return out
  return out

def aic_get_title(title: str, artist: Optional[str] = None, timeout: int = 25):
  try:
    jr = requests.get(
      AIC_SEARCH,
      params={
        "q": title,
        "limit": 50,
        "fields": "id,title,artist_title,date_display,image_id,is_public_domain",
        "query[exists][field]": "image_id",
      },
      timeout=timeout
    ).json()
    data = jr.get("data", [])
    cfg = jr.get("config", {}) or {}
    iiif = cfg.get("iiif_url", "https://www.artic.edu/iiif/2")
    for it in data:
      if not it.get("is_public_domain"): continue
      if artist and not _infix(artist, it.get("artist_title", "")): continue
      img_id = it.get("image_id")
      if not img_id: continue
      url = f"{iiif}/{img_id}/full/843,/0/default.jpg"
      return {
        "id": f"aic_{it['id']}",
        "title": it.get("title"),
        "artist": it.get("artist_title"),
        "year": it.get("date_display"),
        "image_url": url,
        "source": "aic",
      }
  except Exception:
    pass
  return None

def aic_get_by_artist(artist: str, want: int = 5, timeout: int = 25) -> List[Dict]:
  out: List[Dict] = []
  try:
    jr = requests.get(
      AIC_SEARCH,
      params={
        "q": artist,
        "limit": 100,
        "fields": "id,title,artist_title,date_display,image_id,is_public_domain",
        "query[exists][field]": "image_id",
        "query[term][is_public_domain]": "true",
      },
      timeout=timeout
    ).json()
    data = jr.get("data", [])
    cfg = jr.get("config", {}) or {}
    iiif = cfg.get("iiif_url", "https://www.artic.edu/iiif/2")
    for it in data:
      if not it.get("is_public_domain"): continue
      if not _infix(artist, it.get("artist_title","")): continue
      img_id = it.get("image_id")
      if not img_id: continue
      url = f"{iiif}/{img_id}/full/843,/0/default.jpg"
      out.append({
        "id": f"aic_{it['id']}",
        "title": it.get("title"),
        "artist": it.get("artist_title"),
        "year": it.get("date_display"),
        "image_url": url,
        "source": "aic",
      })
      if len(out) >= want: break
  except Exception:
    return out
  return out

# ------------------------------ main ------------------------------------
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="backend/models/artworks_seed_120.json")
  ap.add_argument("--target", type=int, default=120)
  ap.add_argument("--per-artist", dest="per_artist", type=int, default=6,
                  help="how many per artist in the fill phase")
  ap.add_argument("--sleep", type=float, default=0.15, help="seconds between API calls")
  args = ap.parse_args()

  picked: List[Dict] = []
  seen_ids = set()

  def maybe_add(item: Optional[Dict]) -> bool:
    if not item: return False
    if item["id"] in seen_ids: return False
    picked.append({
      "id": item["id"],
      "title": item.get("title"),
      "artist": item.get("artist"),
      "year": item.get("year"),
      "image_url": item.get("image_url"),
    })
    seen_ids.add(item["id"])
    return True

  print("[*] Phase 1: strict title+artist", flush=True)
  for q in FAMOUS_TITLES:
    if len(picked) >= args.target: break
    title, artist = q["title"], q.get("artist")
    item = met_get_title(title, artist) or aic_get_title(title, artist)
    if maybe_add(item):
      print(f"  [+] {item['source'].upper()}: {item['title']} — {item.get('artist','?')}", flush=True)
    else:
      print(f"  [skip] {title} — {artist or ''}", flush=True)
    time.sleep(args.sleep)

  if len(picked) < args.target:
    print("[*] Phase 2: fill by artist", flush=True)
    for artist in FAMOUS_ARTISTS:
      if len(picked) >= args.target: break
      added_here = 0

      # Met first
      try:
        for it in met_get_by_artist(artist, want=args.per_artist):
          if len(picked) >= args.target: break
          if maybe_add(it):
            print(f"  [+] Met: {it['title']} — {artist}", flush=True)
            added_here += 1
            time.sleep(args.sleep)
      except Exception:
        pass

      # AIC next (only if still need more)
      if added_here < args.per_artist and len(picked) < args.target:
        try:
          need = args.per_artist - added_here
          for it in aic_get_by_artist(artist, want=need):
            if len(picked) >= args.target: break
            if maybe_add(it):
              print(f"  [+] AIC: {it['title']} — {artist}", flush=True)
              time.sleep(args.sleep)
        except Exception:
          pass

  if not picked:
    raise SystemExit("No artworks found. Check your network or try a smaller target.")

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, "w", encoding="utf-8") as f:
    json.dump(picked, f, ensure_ascii=False, indent=2)
  print(f"[DONE] Wrote {len(picked)} items → {args.out}", flush=True)

if __name__ == "__main__":
  main()


