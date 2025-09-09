# backend/db_mongo.py
from __future__ import annotations

import os
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
DB_NAME = os.getenv("MONGODB_DB", "artspot")
COLL_NAME = os.getenv("MONGODB_COLL", "artwork_vectors")
USE_ATLAS = os.getenv("USE_ATLAS_VECTOR", "true").lower() == "true"
ATLAS_INDEX_NAME = os.getenv("ATLAS_INDEX_NAME", "artvec")

_client: Optional[MongoClient] = None
_coll: Optional[Collection] = None


def get_db_and_coll() -> Tuple[MongoClient, Collection]:
    global _client, _coll
    if _coll is None:
        _client = MongoClient(MONGODB_URI)
        _coll = _client[DB_NAME][COLL_NAME]
    return _client, _coll


def insert_vectors(docs: List[Dict[str, Any]]) -> int:
    """
    Insert many vector docs.

    Each doc must include:
      artwork_id (str), title (str|None), artist (str|None), year (str|None),
      image_url (str), embedding (List[float] of length 512, L2-normalized)
    """
    if not docs:
        return 0
    _, coll = get_db_and_coll()
    coll.insert_many(docs)
    return len(docs)


def search_vector(
    vec: np.ndarray,
    limit: int = 25,
    num_candidates: int = 200,
) -> List[Dict[str, Any]]:
    """
    Search by vector.

    Preferred path: Atlas Vector Search ($vectorSearch) → returns 'score' (higher is better).
    Fallback: brute-force cosine in Python with the same 'score' field.

    Returns:
      [{ artwork_id, title, artist, year, image_url, score }, ...]
    """
    _, coll = get_db_and_coll()
    vlist = vec.astype("float32").tolist()

    if USE_ATLAS:
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": ATLAS_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": vlist,
                        "numCandidates": num_candidates,
                        "limit": limit,
                        "similarity": "cosine"
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "artwork_id": 1,
                        "title": 1,
                        "artist": 1,
                        "year": 1,
                        "image_url": 1,
                        "score": { "$meta": "vectorSearchScore" }
                    }
                }
            ]
            return list(coll.aggregate(pipeline, allowDiskUse=True))
        except OperationFailure:
            # Atlas not available or index missing → fall through
            pass
        except Exception:
            pass

    # ---- Brute-force cosine fallback (OK for small/medium collections) ----
    cur = coll.find(
        {},
        {
            "_id": 0,
            "artwork_id": 1, "title": 1, "artist": 1, "year": 1,
            "image_url": 1, "embedding": 1
        }
    )
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []
    for d in cur:
        e = d.get("embedding")
        if not e:
            continue
        embs.append(e)
        metas.append({
            "artwork_id": d["artwork_id"],
            "title": d.get("title"),
            "artist": d.get("artist"),
            "year": d.get("year"),
            "image_url": d.get("image_url")
        })
    if not embs:
        return []

    M = np.asarray(embs, dtype="float32")          # (N,512), assumed normalized
    q = vec.reshape(1, -1).astype("float32")       # (1,512)
    sims = (M @ q.T).ravel()                       # cosine similarity
    idxs = np.argsort(-sims)[:limit]               # higher is better
    out: List[Dict[str, Any]] = []
    for i in idxs:
        m = metas[i].copy()
        m["score"] = float(sims[i])
        out.append(m)
    return out


def health_check() -> Dict[str, Any]:
    """Quick connection check for debugging."""
    client, coll = get_db_and_coll()
    return {
        "ok": True,
        "server_info": client.server_info().get("version", "?"),
        "db": DB_NAME,
        "collection": COLL_NAME,
        "use_atlas": USE_ATLAS,
        "atlas_index": ATLAS_INDEX_NAME,
        "count": coll.estimated_document_count(),
    }

