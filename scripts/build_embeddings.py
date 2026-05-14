"""
Build sentence-transformers embeddings for each catalog row.

Writes:
  - app/data/embeddings.npy
  - app/data/catalog_meta.json (urls + model metadata; row order matches embeddings)

Usage:
  python scripts/build_embeddings.py --catalog app/data/catalog.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings.encoder import MODEL_NAME, encode_texts  # noqa: E402


def build_text(item: dict) -> str:
    name = item.get("name", "")
    desc = item.get("description", "")
    skills = ", ".join(item.get("skills", []) or [])
    roles = ", ".join(item.get("job_roles", []) or [])
    prefs = ", ".join(item.get("assessment_preferences", []) or [])  # usually absent in catalog
    types = " ".join(item.get("test_type_labels", []) or item.get("test_type_codes", []) or [])
    remote = str(item.get("remote_testing_supported", ""))
    duration = str(item.get("duration_minutes", ""))
    return "\n".join(
        [
            f"Title: {name}",
            f"Description: {desc}",
            f"Skills: {skills}",
            f"Job roles: {roles}",
            f"Test types: {types}",
            f"Remote testing: {remote}",
            f"Duration minutes: {duration}",
            f"Extra: {prefs}",
        ]
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("app/data/catalog.json"))
    parser.add_argument("--out-emb", type=Path, default=Path("app/data/embeddings.npy"))
    parser.add_argument("--out-meta", type=Path, default=Path("app/data/catalog_meta.json"))
    args = parser.parse_args()

    items = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    texts = [build_text(i) for i in items]
    urls = [str(i["url"]) for i in items]

    emb = encode_texts(texts, batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)

    args.out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_emb, emb)

    meta = {
        "model_name": MODEL_NAME,
        "embedding_dim": int(emb.shape[1]),
        "num_items": int(emb.shape[0]),
        "urls": urls,
    }
    args.out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved embeddings {emb.shape} to {args.out_emb}")
    print(f"Saved meta ({meta['num_items']} urls) to {args.out_meta}")


if __name__ == "__main__":
    main()
