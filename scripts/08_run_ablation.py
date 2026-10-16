"""Script 08: Run ablation experiments across configurations A0-A7."""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import yaml
from langgraph.checkpoint.memory import MemorySaver

from legalrag.core.config import load_config
from legalrag.core.logging import setup_logging
from legalrag.data.db import get_engine
from legalrag.evaluation.citation_metrics import (
    compute_citation_accuracy,
    compute_citation_support,
)
from legalrag.evaluation.datasets import load_golden_set
from legalrag.evaluation.generation_metrics import (
    compute_refusal_accuracy,
    compute_validity_accuracy,
)
from legalrag.evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics_for_query,
)
from legalrag.graph.build import build_graph
from legalrag.graph.deps import GraphDeps
from legalrag.llm.factory import LLMFactory, get_llm
from legalrag.pipelines.simple_rag import run_simple_rag
from legalrag.retrieval.embedder import FakeEmbedder, SentenceTransformerEmbedder
from legalrag.retrieval.reranker import FakeReranker, VietnameseCrossEncoderReranker
from legalrag.retrieval.retriever import Retriever
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import get_qdrant_client
from legalrag.validity.checker import ValidityChecker
from legalrag.validity.repository import ValidityRepository


def apply_overrides(target: dict, source: dict) -> None:
    """Recursively apply overrides into a dictionary."""
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            apply_overrides(target[k], v)
        else:
            target[k] = v


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG ablation experiments")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML (e.g. configs/experiments/A1.yaml)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="Dataset split to evaluate on")
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl", help="Path to golden set")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    args = parser.parse_args()

    setup_logging(level="WARNING")
    config_bundle = load_config()
    app_cfg = config_bundle.app
    models_cfg = config_bundle.models
    retrieval_cfg = config_bundle.retrieval

    exp_path = Path(args.config)
    if not exp_path.exists():
        print(f"Error: Experiment config not found at {exp_path}")
        sys.exit(1)

    with open(exp_path, "r", encoding="utf-8") as f:
        exp_dict = yaml.safe_load(f)

    exp_name = exp_dict.get("name", "exp")
    pipeline_type = exp_dict.get("pipeline", "simple_rag")
    overrides = exp_dict.get("overrides", {})

    print(f"--- Running Ablation Experiment: {exp_name} ({exp_dict.get('description', '')}) ---")
    print(f"Pipeline: {pipeline_type} | Split: {args.split}")

    # Apply retrieval overrides
    ret_overrides = overrides.get("retrieval", {})
    hybrid_enabled = ret_overrides.get("hybrid", {}).get("enabled", retrieval_cfg.hybrid.enabled)
    rerank_enabled = ret_overrides.get("rerank", {}).get("enabled", retrieval_cfg.rerank.enabled)
    parent_child_enabled = ret_overrides.get("parent_child", {}).get("enabled", retrieval_cfg.parent_child.enabled)
    expired_policy = ret_overrides.get("expired_policy", retrieval_cfg.expired_policy)

    # Apply app/graph/validity overrides
    app_overrides = overrides.get("app", {})
    if "graph" in app_overrides:
        g_ov = app_overrides["graph"]
        if "mode" in g_ov:
            app_cfg.graph.mode = g_ov["mode"]
        if "grade" in g_ov and "mode" in g_ov["grade"]:
            app_cfg.graph.grade.mode = g_ov["grade"]["mode"]
        if "verify" in g_ov and "mode" in g_ov["verify"]:
            app_cfg.graph.verify.mode = g_ov["verify"]["mode"]
        if "multi_step" in g_ov and "enabled" in g_ov["multi_step"]:
            app_cfg.graph.multi_step.enabled = g_ov["multi_step"]["enabled"]

    if "validity" in app_overrides:
        v_ov = app_overrides["validity"]
        if "enabled" in v_ov:
            app_cfg.validity.enabled = v_ov["enabled"]
        if "expand_replacements" in v_ov:
            app_cfg.validity.expand_replacements = v_ov["expand_replacements"]

    golden_items = load_golden_set(args.golden_set)
    eval_items = [it for it in golden_items if it.split == args.split]
    if args.limit:
        eval_items = eval_items[:args.limit]

    print(f"Evaluating {len(eval_items)} questions...")

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
        reranker = FakeReranker() if rerank_enabled else None
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
        ) if rerank_enabled else None

    bm25 = SyllableBM25(mode="syllable_bigram") if hybrid_enabled else None

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
        hybrid_enabled=hybrid_enabled,
        rerank_enabled=rerank_enabled,
        rerank_top_k=retrieval_cfg.rerank.top_k,
        parent_child_enabled=parent_child_enabled,
        max_parents=10,
        expired_policy=expired_policy,
        status_penalty=retrieval_cfg.status_penalty,
        context_token_budget=retrieval_cfg.context_token_budget,
    )

    llm = get_llm("generator", config_bundle=config_bundle)

    # Initialize graph if pipeline is graph
    compiled_graph = None
    if pipeline_type == "graph":
        validity_repo = ValidityRepository(engine)
        validity_checker = ValidityChecker(validity_repo, snapshot_as_of=app_cfg.snapshot_as_of)
        llm_factory = LLMFactory(config_bundle)
        deps = GraphDeps(
            config=config_bundle,
            retriever=retriever,
            validity_checker=validity_checker,
            llm_factory=llm_factory,
        )
        compiled_graph = build_graph(deps, checkpointer=MemorySaver())

    query_ret_metrics: list[dict[str, float]] = []
    citation_accuracies: list[float] = []
    citation_supports: list[float] = []
    refusal_accuracies: list[float] = []
    validity_accuracies: list[float] = []
    latencies: list[float] = []
    llm_calls_total = 0

    for it in eval_items:
        t0 = time.time()
        if pipeline_type == "graph" and compiled_graph:
            thread_id = str(uuid.uuid4())
            inputs = {
                "question": it.question,
                "thread_id": thread_id,
                "has_upload": False,
                "messages": it.history or [],
            }
            res = compiled_graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
            answer = res.get("answer", "")
            contexts = res.get("contexts", [])
            citations = res.get("citations", [])
            warnings = res.get("warnings", [])
            llm_c = res.get("llm_calls", 1)
            grade_res = res.get("grade")
            insufficient = (grade_res == "insufficient" or "không đủ căn cứ" in answer.lower())
        else:
            output = run_simple_rag(
                query=it.question,
                retriever=retriever,
                llm=llm,
                max_parents=5,
            )
            answer = output.answer
            contexts = output.contexts
            citations = output.citations
            warnings = []
            llm_c = 1
            insufficient = output.insufficient_context

        elapsed_ms = (time.time() - t0) * 1000.0
        latencies.append(elapsed_ms)
        llm_calls_total += llm_c

        if it.gold_articles:
            rm = compute_retrieval_metrics_for_query(
                retrieved=contexts,
                gold_articles=it.gold_articles,
                k_list=(5, 10),
            )
            query_ret_metrics.append(rm)

            c_acc = compute_citation_accuracy(citations, it.gold_articles)
            citation_accuracies.append(c_acc)

            c_sup = compute_citation_support(answer, citations, contexts)
            citation_supports.append(c_sup)

        r_acc = compute_refusal_accuracy(
            answer=answer,
            expected_behavior=it.expected_behavior,
            insufficient_context=insufficient,
        )
        refusal_accuracies.append(r_acc)

        v_acc = compute_validity_accuracy(warnings, it.expected_validity)
        validity_accuracies.append(v_acc)

    # Aggregations
    agg_ret = aggregate_retrieval_metrics(query_ret_metrics)
    mean_cit_acc = float(np.mean(citation_accuracies)) if citation_accuracies else 0.0
    mean_cit_sup = float(np.mean(citation_supports)) if citation_supports else 0.0
    mean_ref_acc = float(np.mean(refusal_accuracies)) if refusal_accuracies else 0.0
    mean_val_acc = float(np.mean(validity_accuracies)) if validity_accuracies else 0.0

    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    avg_llm_calls = float(llm_calls_total / len(eval_items)) if eval_items else 0.0

    summary = {
        "experiment": exp_name,
        "description": exp_dict.get("description", ""),
        "pipeline": pipeline_type,
        "split": args.split,
        "num_evaluated": len(eval_items),
        "recall@5": round(agg_ret.get("recall@5", 0.0), 4),
        "recall@10": round(agg_ret.get("recall@10", 0.0), 4),
        "mrr@10": round(agg_ret.get("mrr@10", 0.0), 4),
        "ndcg@10": round(agg_ret.get("ndcg@10", 0.0), 4),
        "citation_accuracy": round(mean_cit_acc, 4),
        "citation_support": round(mean_cit_sup, 4),
        "refusal_accuracy": round(mean_ref_acc, 4),
        "validity_accuracy": round(mean_val_acc, 4),
        "avg_llm_calls": round(avg_llm_calls, 2),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "latency_mean_ms": round(mean_lat, 2),
    }

    print("\n" + "=" * 50)
    print(f"--- ABLATION RESULTS: {exp_name} ---")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print("=" * 50)

    # Save result
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{timestamp}_{exp_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved result to {out_file}")


if __name__ == "__main__":
    main()
