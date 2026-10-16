"""Script 04: Build Qdrant dense+sparse vector index and verify SQL tables."""

import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from legalrag.core.config import load_app_config, load_models_config, load_retrieval_config
from legalrag.core.logging import setup_logging
from legalrag.ingestion.indexer import index_chunks
from legalrag.retrieval.embedder import FakeEmbedder
from legalrag.retrieval.store import get_qdrant_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant vector index for Legal RAG")
    parser.add_argument("--chunks", default="data/processed/chunks.parquet", help="Path to chunks.parquet")
    parser.add_argument("--embeddings", default="data/processed/embeddings.parquet", help="Path to embeddings.parquet")
    parser.add_argument("--collection", default=None, help="Collection name (default from app.yaml)")
    parser.add_argument("--qdrant-path", default=None, help="Qdrant storage directory (default from app.yaml)")
    parser.add_argument("--qdrant-url", default=None, help="Remote Qdrant server URL")
    parser.add_argument("--tag", default=None, help="Tag for index metadata filename")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection from scratch")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for upsert")
    parser.add_argument("--fake-embeddings", action="store_true", help="Use deterministic FakeEmbedder for local development testing")
    args = parser.parse_args()

    setup_logging()
    app_cfg = load_app_config()
    models_cfg = load_models_config()
    retrieval_cfg = load_retrieval_config()

    collection_name = args.collection or app_cfg.qdrant.collection
    qdrant_path = args.qdrant_path or app_cfg.qdrant.path
    qdrant_url = args.qdrant_url or app_cfg.qdrant.url

    chunks_path = Path(args.chunks)
    embeddings_path = Path(args.embeddings)

    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        sys.exit(1)

    print("--- Building Vector Index ---")
    print(f"Collection  : {collection_name}")
    print(f"Qdrant Path : {qdrant_path}")
    print(f"Chunks file : {chunks_path}")

    chunks_df = pd.read_parquet(chunks_path)
    print(f"Loaded {len(chunks_df)} chunks from parquet.")

    # Merge document metadata into chunks_df if available
    docs_path = Path("data/interim/corpus_docs.parquet")
    if docs_path.exists():
        docs_df = pd.read_parquet(docs_path)
        meta_cols = ["doc_id", "title", "doc_number", "doc_type", "status", "effective_date", "url"]
        cols_to_use = [c for c in meta_cols if c in docs_df.columns]
        docs_subset = docs_df[cols_to_use].drop_duplicates(subset=["doc_id"])
        # Drop columns from chunks_df if they already exist to avoid suffix collision
        for col in cols_to_use:
            if col != "doc_id" and col in chunks_df.columns:
                chunks_df = chunks_df.drop(columns=[col])
        chunks_df = chunks_df.merge(docs_subset, on="doc_id", how="left")
        print("Merged document metadata into chunks payload.")

    # Handle embeddings
    embed_cfg = models_cfg.embeddings.get("default", {})
    embed_model_name = getattr(embed_cfg, "model", "AITeamVN/Vietnamese_Embedding")
    dense_dim = getattr(embed_cfg, "dim", 1024)

    embeddings_map: dict[str, list[float]] = {}
    if args.fake_embeddings:
        print("Notice: Generating fake embeddings (dim=64) for local development testing...")
        fake_embedder = FakeEmbedder(dim=64, name="fake-embedder")
        embed_model_name = "fake-embedder"
        dense_dim = 64
        for _, r in chunks_df.iterrows():
            cid = str(r["chunk_id"])
            vec = fake_embedder.encode_query(str(r["embed_text"]))
            embeddings_map[cid] = vec.tolist()
    elif embeddings_path.exists():
        print(f"Reading embeddings from {embeddings_path}...")
        emb_df = pd.read_parquet(embeddings_path)
        for _, r in emb_df.iterrows():
            embeddings_map[str(r["chunk_id"])] = list(r["vector"])
        if embeddings_map:
            sample_vec = next(iter(embeddings_map.values()))
            dense_dim = len(sample_vec)
        print(f"Loaded {len(embeddings_map)} vectors (dim={dense_dim}).")
    else:
        print(f"Error: Embeddings file not found at {embeddings_path}.")
        print("To generate embeddings:")
        print("  Option A (Colab GPU): run notebooks/colab_embed.ipynb and save to data/processed/embeddings.parquet")
        print("  Option B (CPU sample): run python scripts/03_embed_chunks.py --limit 100")
        print("  Option C (Dev / Test): run with --fake-embeddings flag")
        sys.exit(1)

    # Initialize client
    client = get_qdrant_client(url=qdrant_url, path=qdrant_path)

    state_file = Path("data/processed/index_state.parquet")
    meta_name = f"index_meta_{args.tag}.json" if args.tag else "index_meta.json"
    meta_file = Path(f"data/processed/{meta_name}")

    bm25_cfg = getattr(retrieval_cfg, "sparse", None)
    bm25_k1 = getattr(bm25_cfg, "k1", 1.2) if bm25_cfg else 1.2
    bm25_b = getattr(bm25_cfg, "b", 0.75) if bm25_cfg else 0.75

    meta = index_chunks(
        client=client,
        chunks_df=chunks_df,
        embeddings_map=embeddings_map,
        embed_model_name=embed_model_name,
        dense_dim=dense_dim,
        collection_name=collection_name,
        state_file=state_file,
        meta_file=meta_file,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        snapshot_as_of=app_cfg.snapshot_as_of,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )

    print("\n--- Indexing Completed Successfully ---")
    print(f"Model       : {meta['embed_model']} (dim={meta['dim']})")
    print(f"Chunks total: {meta['num_chunks']}")
    print(f"Upserted    : {meta['upserted_count']}")
    print(f"Deleted     : {meta['deleted_count']}")
    print(f"Meta file   : {meta_file}")


if __name__ == "__main__":
    main()
