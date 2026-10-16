"""Script 07: Calibrate grade thresholds (tau_high, tau_low) using dev split."""

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import yaml

from legalrag.core.config import load_config
from legalrag.core.logging import setup_logging
from legalrag.data.db import get_engine
from legalrag.evaluation.datasets import load_golden_set
from legalrag.retrieval.embedder import FakeEmbedder, SentenceTransformerEmbedder
from legalrag.retrieval.reranker import FakeReranker, VietnameseCrossEncoderReranker
from legalrag.retrieval.retriever import Retriever
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import get_qdrant_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate tau_high and tau_low thresholds")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl", help="Path to golden set")
    parser.add_argument("--app-config", default="configs/app.yaml", help="Path to app.yaml to update")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes to app.yaml")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    config_bundle = load_config()
    app_cfg = config_bundle.app
    models_cfg = config_bundle.models
    retrieval_cfg = config_bundle.retrieval

    golden_items = load_golden_set(args.golden_set)
    dev_items = [
        it for it in golden_items
        if it.split == "dev" and it.gold_articles and it.review_status in ["accepted", "edited"]
    ]

    print(f"Loaded {len(dev_items)} dev items with gold targets from {args.golden_set}")
    if not dev_items:
        print("Error: No valid dev items found for calibration.")
        sys.exit(1)

    client = get_qdrant_client(url=app_cfg.qdrant.url, path=app_cfg.qdrant.path)
    engine = get_engine(app_cfg.db.url)

    # Check index_meta to verify embedding model
    meta_path = Path("data/processed/index_meta.json")
    is_fake_index = False
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            imeta = json.load(f)
        if imeta.get("embed_model") == "fake-embedder":
            is_fake_index = True

    if is_fake_index:
        print("Note: Index was built with fake-embedder. Using FakeEmbedder and FakeReranker.")
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

    positive_scores: list[float] = []
    distractor_scores: list[float] = []

    print("Running retrieval and scoring on dev items...")
    for it in dev_items:
        gold_targets = {(g.doc_id, str(g.article_no)) for g in it.gold_articles}
        results = retriever.retrieve(it.question, scope="corpus", max_parents=10)

        for ctx in results:
            art_key = (ctx.doc_id, str(ctx.article_no))
            if art_key in gold_targets:
                positive_scores.append(ctx.best_score)
            else:
                distractor_scores.append(ctx.best_score)

    print(f"Collected {len(positive_scores)} positive scores and {len(distractor_scores)} distractor scores.")

    if positive_scores:
        tau_high = float(round(np.percentile(positive_scores, 50), 3))
    else:
        tau_high = 3.0

    if distractor_scores:
        tau_low = float(round(np.percentile(distractor_scores, 95), 3))
    else:
        tau_low = 0.5

    # Ensure tau_low < tau_high
    if tau_low >= tau_high:
        tau_low = round(tau_high * 0.5, 3)

    print("\n--- Calibration Results ---")
    print(f"tau_high (median positive score) : {tau_high}")
    print(f"tau_low  (p95 distractor score)  : {tau_low}")

    if not args.dry_run:
        config_path = Path(args.app_config)
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)

        raw_cfg.setdefault("graph", {}).setdefault("grade", {})
        raw_cfg["graph"]["grade"]["tau_high"] = tau_high
        raw_cfg["graph"]["grade"]["tau_low"] = tau_low

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(raw_cfg, f, allow_unicode=True, sort_keys=False)

        print(f"Successfully updated {config_path} with calibrated thresholds.")


if __name__ == "__main__":
    main()
