"""Script 03: Embed chunks into dense vectors with resume capability."""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch

from legalrag.core.config import load_models_config
from legalrag.core.logging import setup_logging
from legalrag.retrieval.embedder import SentenceTransformerEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks with SentenceTransformer")
    parser.add_argument("--chunks", default="data/processed/chunks.parquet", help="Path to chunks.parquet")
    parser.add_argument("--out", default="data/processed/embeddings.parquet", help="Path to output embeddings.parquet")
    parser.add_argument("--model", default=None, help="Embedding model name (default from models.yaml)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to embed (for testing)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--device", default=None, help="Inference device: cpu or cuda")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume, re-embed all")
    args = parser.parse_args()

    setup_logging()
    chunks_path = Path(args.chunks)
    out_path = Path(args.out)

    if not chunks_path.exists():
        print(f"Error: chunks file not found: {chunks_path}")
        sys.exit(1)

    models_cfg = load_models_config()
    embed_cfg = models_cfg.embeddings.get("default", {})
    model_name = args.model or getattr(embed_cfg, "model", "AITeamVN/Vietnamese_Embedding")
    max_length = getattr(embed_cfg, "max_length", 2048)
    query_prefix = getattr(embed_cfg, "query_prefix", "")
    doc_prefix = getattr(embed_cfg, "doc_prefix", "")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("--- Embedding Chunks ---")
    print(f"Chunks file : {chunks_path}")
    print(f"Output file : {out_path}")
    print(f"Model       : {model_name}")
    print(f"Device      : {device}")
    print(f"Batch size  : {args.batch_size}")

    chunks_df = pd.read_parquet(chunks_path)
    total_in_file = len(chunks_df)
    if args.limit:
        chunks_df = chunks_df.head(args.limit)
    print(f"Total chunks to process: {len(chunks_df)} (out of {total_in_file})")

    existing_embeddings: dict[str, list[float]] = {}
    if not args.no_resume and out_path.exists():
        try:
            prev_df = pd.read_parquet(out_path)
            for _, r in prev_df.iterrows():
                existing_embeddings[str(r["chunk_id"])] = list(r["vector"])
            print(f"Resuming: found {len(existing_embeddings)} already embedded chunks in {out_path}")
        except Exception as e:
            print(f"Warning: could not read existing embeddings for resume: {e}")

    # Determine chunks that need embedding
    to_embed_rows = [
        row for _, row in chunks_df.iterrows()
        if str(row["chunk_id"]) not in existing_embeddings
    ]
    print(f"Chunks remaining to embed: {len(to_embed_rows)}")

    if to_embed_rows:
        embedder = SentenceTransformerEmbedder(
            model_name=model_name,
            max_seq_length=max_length,
            device=device,
            query_prefix=query_prefix,
            doc_prefix=doc_prefix,
        )

        texts = [str(r["embed_text"]) for r in to_embed_rows]
        cids = [str(r["chunk_id"]) for r in to_embed_rows]

        start_time = time.time()
        print(f"Starting embedding on {len(texts)} chunks...")

        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            batch_cids = cids[i : i + args.batch_size]
            batch_vecs = embedder.encode_documents(batch_texts, batch_size=len(batch_texts))
            for cid, vec in zip(batch_cids, batch_vecs, strict=False):
                existing_embeddings[cid] = vec.tolist()

            done = min(i + args.batch_size, len(texts))
            elapsed = time.time() - start_time
            rate = done / max(elapsed, 0.001)
            print(f"  Processed {done}/{len(texts)} chunks ({rate:.1f} chunks/s) - Elapsed: {elapsed:.1f}s")

    # Save to parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_rows = [
        {"chunk_id": cid, "vector": vec}
        for cid, vec in existing_embeddings.items()
        if cid in set(chunks_df["chunk_id"].astype(str))
    ]
    pd.DataFrame(out_rows).to_parquet(out_path, index=False)
    print(f"Saved {len(out_rows)} embeddings to {out_path}")


if __name__ == "__main__":
    main()
