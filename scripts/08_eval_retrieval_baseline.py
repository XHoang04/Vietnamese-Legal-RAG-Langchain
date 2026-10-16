"""Run baseline retrieval evaluation on dev and test splits, and save results."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from legalrag.core.config import load_config
from legalrag.core.logging import setup_logging
from legalrag.data.db import get_engine
from legalrag.evaluation.datasets import load_golden_set
from legalrag.evaluation.runner import evaluate_retrieval
from legalrag.retrieval.embedder import FakeEmbedder, SentenceTransformerEmbedder
from legalrag.retrieval.reranker import FakeReranker, VietnameseCrossEncoderReranker
from legalrag.retrieval.retriever import Retriever
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import get_qdrant_client


def main() -> None:
    setup_logging(level="WARNING")
    config_bundle = load_config()
    app_cfg = config_bundle.app
    models_cfg = config_bundle.models
    retrieval_cfg = config_bundle.retrieval

    golden_items = load_golden_set("eval/golden_set.jsonl")
    dev_items = [it for it in golden_items if it.split == "dev"]
    test_items = [it for it in golden_items if it.split == "test"]

    print(f"Loaded Golden Set: {len(golden_items)} items (dev={len(dev_items)}, test={len(test_items)})")

    client = get_qdrant_client(url=app_cfg.qdrant.url, path=app_cfg.qdrant.path)
    engine = get_engine(app_cfg.db.url)

    meta_path = Path("data/processed/index_meta.json")
    is_fake_index = False
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            imeta = json.load(f)
        if imeta.get("embed_model") == "fake-embedder":
            is_fake_index = True

    if is_fake_index:
        embedder = FakeEmbedder(dim=64, name="fake-embedder")
        reranker = FakeReranker()
    else:
        embed_cfg = models_cfg.embeddings.get("default", {})
        embedder = SentenceTransformerEmbedder(
            model_name=getattr(embed_cfg, "model", "AITeamVN/Vietnamese_Embedding"),
            max_seq_length=getattr(embed_cfg, "max_length", 2048),
            device="cpu",
        )
        rerank_cfg = models_cfg.reranker.get("default", {})
        reranker = VietnameseCrossEncoderReranker(
            model_name=getattr(rerank_cfg, "model", "AITeamVN/Vietnamese_Reranker"),
            max_length=getattr(rerank_cfg, "max_length", 1024),
            device="cpu",
        )

    bm25 = SyllableBM25(mode="syllable_bigram")

    retriever = Retriever(
        client=client,
        embedder=embedder,
        bm25=bm25,
        reranker=reranker,
        db_engine=engine,
        collection_name=app_cfg.qdrant.collection,
        dense_top_k=retrieval_cfg.dense_top_k,
        sparse_top_k=retrieval_cfg.sparse_top_k,
        rrf_k=retrieval_cfg.rrf_k,
        fused_top_k=retrieval_cfg.fused_top_k,
        hybrid_enabled=retrieval_cfg.hybrid.enabled,
        rerank_enabled=retrieval_cfg.rerank.enabled,
        rerank_top_k=retrieval_cfg.rerank.top_k,
        parent_child_enabled=retrieval_cfg.parent_child.enabled,
        max_parents=10,
        expired_policy=retrieval_cfg.expired_policy,
        status_penalty=retrieval_cfg.status_penalty,
        context_token_budget=retrieval_cfg.context_token_budget,
    )

    print("\n--- Evaluating DEV Split ---")
    dev_results = evaluate_retrieval(retriever, dev_items, max_parents=10)
    print(f"Dev Overall Metrics : {dev_results['overall']}")
    print(f"Dev Latency (ms)    : {dev_results['latency_ms']}")

    print("\n--- Evaluating TEST Split ---")
    test_results = evaluate_retrieval(retriever, test_items, max_parents=10)
    print(f"Test Overall Metrics: {test_results['overall']}")
    print(f"Test Latency (ms)   : {test_results['latency_ms']}")

    # Save to eval/results/
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{timestamp}_baseline_retrieval.json"

    payload = {
        "timestamp": timestamp,
        "config": {
            "hybrid_enabled": retrieval_cfg.hybrid.enabled,
            "rerank_enabled": retrieval_cfg.rerank.enabled,
            "parent_child_enabled": retrieval_cfg.parent_child.enabled,
            "embed_model": embedder.name,
        },
        "dev": dev_results,
        "test": test_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved baseline evaluation results to {out_file}")


if __name__ == "__main__":
    main()
