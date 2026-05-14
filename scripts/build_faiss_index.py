"""
Build a FAISS inner-product index from embeddings.npy (L2-normalized vectors).

Usage:
  python scripts/build_faiss_index.py --embeddings app/data/embeddings.npy --out app/data/faiss.index
"""

from __future__ import annotations

import argparse

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, default="app/data/embeddings.npy")
    parser.add_argument("--out", type=str, default="app/data/faiss.index")
    args = parser.parse_args()

    emb = np.load(args.embeddings).astype("float32", copy=False)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    faiss.write_index(index, args.out)
    print(f"Wrote FAISS index nt={index.ntotal}, dim={d} -> {args.out}")


if __name__ == "__main__":
    main()
