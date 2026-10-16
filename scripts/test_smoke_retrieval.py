"""Evaluate retrieval top-5 hit rate and latency on smoke_queries.yaml."""

import sys
import time
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

    smoke_file = Path("eval/smoke_queries.yaml")
    if not smoke_file.exists():
        print(f"Error: {smoke_file} not found")
        sys.exit(1)

    with open(smoke_file, encoding="utf-8") as f:
        queries = yaml.safe_load(f)

    print(f"Loaded {len(queries)} smoke queries from {smoke_file}\n")

    client = get_qdrant_client(url=app_cfg.qdrant.url, path=app_cfg.qdrant.path)
    engine = get_engine(app_cfg.db.url)

    # Determine embedder type from index_meta.json
    meta_path = Path("data/processed/index_meta.json")
    is_fake_index = False
    if meta_path.exists():
        import json
        with open(meta_path, encoding="utf-8") as f:
            imeta = json.load(f)
        if imeta.get("embed_model") == "fake-embedder":
            is_fake_index = True

    if is_fake_index:
        print("Note: Index was built with fake-embedder. Using FakeEmbedder + BM25 sparse.")
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
        max_parents=5,
        expired_policy=retrieval_cfg.expired_policy,
        status_penalty=retrieval_cfg.status_penalty,
        context_token_budget=retrieval_cfg.context_token_budget,
    )

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    latencies = []

    print(f"{'ID':<6} | {'Target Doc':<16} | {'Art':<4} | {'Top-5 Hit':<10} | {'Rank':<5} | {'Latency':<8} | {'Query'}")
    print("-" * 90)

    for q in queries:
        qid = q["id"]
        qtext = q["query"]
        exp_doc = q["expected_doc_number"]
        exp_art = str(q["expected_article_no"])

        t0 = time.time()
        results = retriever.retrieve(qtext, scope="corpus", max_parents=5)
        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)

        hit_rank = None
        for rank, ctx in enumerate(results, start=1):
            doc_matched = (ctx.doc_number == exp_doc) or (exp_doc in ctx.title)
            art_matched = (str(ctx.article_no) == exp_art)
            if doc_matched and art_matched:
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank == 1:
                hits_at_1 += 1
            if hit_rank <= 3:
                hits_at_3 += 1
            if hit_rank <= 5:
                hits_at_5 += 1
            hit_str = "YES"
            rank_str = str(hit_rank)
        else:
            hit_str = "NO"
            rank_str = "-"

        print(f"{qid:<6} | {exp_doc:<16} | {exp_art:<4} | {hit_str:<10} | {rank_str:<5} | {elapsed_ms:>6.1f}ms | {qtext[:40]}")

    n = len(queries)
    hit_rate_5 = (hits_at_5 / n) * 100.0
    hit_rate_3 = (hits_at_3 / n) * 100.0
    hit_rate_1 = (hits_at_1 / n) * 100.0

    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    mean_lat = np.mean(latencies)

    print("\n" + "=" * 50)
    print("--- SMOKE RETRIEVAL EVALUATION REPORT ---")
    print(f"Total Queries : {n}")
    print(f"Hit@1 Rate    : {hit_rate_1:.1f}% ({hits_at_1}/{n})")
    print(f"Hit@3 Rate    : {hit_rate_3:.1f}% ({hits_at_3}/{n})")
    print(f"Hit@5 Rate    : {hit_rate_5:.1f}% ({hits_at_5}/{n})")
    print(f"Latency mean  : {mean_lat:.1f}ms")
    print(f"Latency p50   : {p50:.1f}ms")
    print(f"Latency p90   : {p90:.1f}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
