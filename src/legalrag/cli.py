"""CLI interface for Legal RAG."""

import argparse
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from legalrag.core.config import load_config
from legalrag.core.logging import setup_logging
from legalrag.data.db import get_engine
from legalrag.data.models import Status
from legalrag.graph.build import build_graph
from legalrag.graph.deps import GraphDeps
from legalrag.llm.factory import LLMFactory
from legalrag.llm.fakes import FakeLLMFactory
from legalrag.retrieval.embedder import FakeEmbedder, SentenceTransformerEmbedder
from legalrag.retrieval.reranker import FakeReranker, VietnameseCrossEncoderReranker
from legalrag.retrieval.retriever import FakeRetriever, RetrievedContext, Retriever
from legalrag.retrieval.sparse import SyllableBM25
from legalrag.retrieval.store import get_qdrant_client
from legalrag.validity.checker import FakeValidityChecker, ValidityChecker, ValidityInfo
from legalrag.validity.repository import ValidityRepository


def ask_fake(question: str) -> None:
    """Execute query through walking skeleton graph with fake dependencies."""
    config_bundle = load_config()
    # For fake walking skeleton, set tau_high so threshold grade passes
    config_bundle.app.graph.grade.tau_high = 0.0

    mock_contexts = [
        RetrievedContext(
            parent_id="doc1:A5",
            doc_id="doc1",
            title="Bộ luật Lao động 2019",
            doc_number="45/2019/QH14",
            breadcrumb="Bộ luật Lao động 2019 > Chương I > Điều 5. Quyền và nghĩa vụ của người lao động",
            article_no="5",
            text="Điều 5. Quyền và nghĩa vụ của người lao động:\n1. Người lao động có các quyền sau đây: làm việc, tự do lựa chọn việc làm...",
            best_score=5.5,
            source="corpus",
            status=Status.IN_FORCE,
            url="http://vbpl.vn/laodong/Pages/vbpq-toanvan.aspx?ItemID=1",
        )
    ]

    mock_llm_responses = {
        "router": [
            '{"intent": "legal_query", "scope": "corpus", "complexity": "simple", "standalone_question": "'
            + question
            + '", "clarifying_question": null}'
        ],
        "generator": [
            '{"answer": "Theo Điều 5 [1], người lao động có quyền tự do lựa chọn việc làm, nghề nghiệp và làm việc cho bất kỳ người sử dụng lao động nào.", "used_ctx": [1], "insufficient_context": false}'
        ],
    }

    deps = GraphDeps(
        config=config_bundle,
        retriever=FakeRetriever(mock_contexts),
        validity_checker=FakeValidityChecker({
            "doc1": ValidityInfo(doc_id="doc1", status=Status.IN_FORCE, warnings=[])
        }),
        llm_factory=FakeLLMFactory(mock_llm_responses),
    )

    graph = build_graph(deps, checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    inputs = {
        "question": question,
        "thread_id": thread_id,
        "has_upload": False,
        "messages": [],
    }

    result = graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})

    print("=== CÂU TRẢ LỜI ===")
    print(result.get("answer", ""))
    print("\n=== DANH SÁCH TRACE ===")
    for evt in result.get("trace", []):
        note = f" ({evt.note})" if evt.note else ""
        print(f"[{evt.node}] {evt.ms:.2f}ms{note}")


def ask_real(question: str) -> None:
    """Execute query through LangGraph with real retriever, DB, and validity checker."""
    config_bundle = load_config()
    app_cfg = config_bundle.app
    models_cfg = config_bundle.models
    retrieval_cfg = config_bundle.retrieval

    client = get_qdrant_client(url=app_cfg.qdrant.url, path=app_cfg.qdrant.path)
    engine = get_engine(app_cfg.db.url)

    meta_path = Path("data/processed/index_meta.json")
    is_fake_index = False
    if meta_path.exists():
        import json
        with open(meta_path, encoding="utf-8") as f:
            imeta = json.load(f)
        if imeta.get("embed_model") == "fake-embedder":
            is_fake_index = True

    if is_fake_index:
        embedder = FakeEmbedder(dim=64, name="fake-embedder")
        reranker = FakeReranker() if retrieval_cfg.rerank.enabled else None
    else:
        embed_cfg = models_cfg.embeddings.get("default", {})
        embedder = SentenceTransformerEmbedder(
            model_name=getattr(embed_cfg, "model", "AITeamVN/Vietnamese_Embedding"),
            max_seq_length=getattr(embed_cfg, "max_length", 2048),
            device="cpu",
        )
        rerank_cfg = models_cfg.reranker.get("default", {})
        reranker = (
            VietnameseCrossEncoderReranker(
                model_name=getattr(rerank_cfg, "model", "AITeamVN/Vietnamese_Reranker"),
                max_length=getattr(rerank_cfg, "max_length", 1024),
                device="cpu",
            )
            if retrieval_cfg.rerank.enabled
            else None
        )

    bm25 = SyllableBM25(mode="syllable_bigram") if retrieval_cfg.hybrid.enabled else None

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

    validity_repo = ValidityRepository(engine)
    validity_checker = ValidityChecker(validity_repo, snapshot_as_of=app_cfg.snapshot_as_of)
    llm_factory = LLMFactory(config_bundle)

    deps = GraphDeps(
        config=config_bundle,
        retriever=retriever,
        validity_checker=validity_checker,
        llm_factory=llm_factory,
    )

    graph = build_graph(deps, checkpointer=MemorySaver())

    thread_id = str(uuid.uuid4())
    inputs = {
        "question": question,
        "thread_id": thread_id,
        "has_upload": False,
        "messages": [],
    }

    result = graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})

    print("=== CÂU TRẢ LỜI ===")
    print(result.get("answer", ""))
    print("\n=== DANH SÁCH TRACE ===")
    for evt in result.get("trace", []):
        note = f" ({evt.note})" if evt.note else ""
        print(f"[{evt.node}] {evt.ms:.2f}ms{note}")


def run_retrieval(
    query: str,
    top_k: int = 5,
    scope: str = "corpus",
    fake: bool = False,
) -> None:
    """Execute hybrid retrieval for a query and print ranked parent contexts."""
    config_bundle = load_config()
    app_cfg = config_bundle.app
    models_cfg = config_bundle.models
    retrieval_cfg = config_bundle.retrieval

    if fake:
        print(f"Executing retrieval in FAKE mode for query: '{query}'")
        retriever = FakeRetriever([
            RetrievedContext(
                parent_id="fake:A105",
                doc_id="fake_doc",
                title="Bộ luật Lao động 2019",
                doc_number="45/2019/QH14",
                breadcrumb="Bộ luật Lao động 2019 > Điều 105. Thời giờ làm việc bình thường",
                article_no="105",
                text="1. Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần.",
                best_score=9.2,
                source="corpus",
                status=Status.IN_FORCE,
            )
        ])
    else:
        qdrant_client = get_qdrant_client(url=app_cfg.qdrant.url, path=app_cfg.qdrant.path)
        embed_cfg = models_cfg.embeddings.get("default", {})
        model_name = getattr(embed_cfg, "model", "AITeamVN/Vietnamese_Embedding")

        # Check index_meta to verify embedding model
        meta_path = Path("data/processed/index_meta.json")
        is_fake_index = False
        if meta_path.exists():
            import json
            with open(meta_path, encoding="utf-8") as f:
                imeta = json.load(f)
            if imeta.get("embed_model") == "fake-embedder":
                is_fake_index = True

        if is_fake_index:
            embedder = FakeEmbedder(dim=64, name="fake-embedder")
            reranker = FakeReranker()
        else:
            embedder = SentenceTransformerEmbedder(
                model_name=model_name,
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
        engine = get_engine(app_cfg.db.url)

        retriever = Retriever(
            client=qdrant_client,
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
            max_parents=top_k,
            expired_policy=retrieval_cfg.expired_policy,
            status_penalty=retrieval_cfg.status_penalty,
            context_token_budget=retrieval_cfg.context_token_budget,
            expected_embed_model=embedder.name,
            index_meta_path=meta_path,
        )

    results = retriever.retrieve(query, scope=scope, max_parents=top_k)

    print(f"\n=== KẾT QUẢ TÌM KIẾM ({len(results)} kết quả) ===")
    print(f"Truy vấn: '{query}'\n")

    if not results:
        print("Không tìm thấy kết quả phù hợp.")
        return

    for rank, ctx in enumerate(results, start=1):
        status_label = ctx.status.value if ctx.status else "UNKNOWN"
        print(f"[{rank}] Điểm: {ctx.best_score:.4f} | Trạng thái: {status_label}")
        print(f"     Văn bản : {ctx.title} (Số hiệu: {ctx.doc_number or 'N/A'})")
        print(f"     Vị trí  : {ctx.breadcrumb}")
        print(f"     Chunks  : {', '.join(ctx.matched_chunk_ids)}")
        preview = ctx.text[:200].replace("\n", " ")
        if len(ctx.text) > 200:
            preview += "..."
        print(f"     Nội dung: {preview}\n")


def run_check_validity(identifier: str) -> None:
    """Check validity status and warnings for a document number or doc_id."""
    config_bundle = load_config()
    db_engine = get_engine(config_bundle.app.db.url)
    repo = ValidityRepository(db_engine)
    checker = ValidityChecker(repo, snapshot_as_of=config_bundle.app.snapshot_as_of)

    doc_id = repo.find_doc_id_by_number(identifier)
    if not doc_id:
        print(f"Không tìm thấy văn bản nào với số hiệu/mã: '{identifier}' trong kho dữ liệu.")
        return

    results = checker.check([doc_id])
    info = results[doc_id]

    print("\n=======================================================")
    print("THÔNG TIN HIỆU LỰC VĂN BẢN")
    print("=======================================================")
    print(f"Doc ID      : {info.doc_id}")
    print(f"Tiêu đề     : {info.title}")
    print(f"Trạng thái  : {info.status.value if info.status else 'UNKNOWN'}")
    print(f"Hiệu lực từ : {info.effective_date or 'N/A'}")
    print(f"Hết hiệu lực: {info.expiry_date or 'N/A'}")

    if info.replaced_by:
        print(f"\nBị thay thế bởi ({len(info.replaced_by)} văn bản):")
        for ref in info.replaced_by:
            url_str = f" ({ref.url})" if ref.url else ""
            print(f" - {ref.title}{url_str}")

    if info.amended_by:
        print(f"\nĐược sửa đổi, bổ sung bởi ({len(info.amended_by)} văn bản):")
        for ref in info.amended_by:
            url_str = f" ({ref.url})" if ref.url else ""
            print(f" - {ref.title}{url_str}")

    if info.partially_repealed_by:
        print(f"\nBị bãi bỏ một phần bởi ({len(info.partially_repealed_by)} văn bản):")
        for ref in info.partially_repealed_by:
            url_str = f" ({ref.url})" if ref.url else ""
            print(f" - {ref.title}{url_str}")

    print("\nCảnh báo hiệu lực:")
    if info.warnings:
        for w in info.warnings:
            print(f" [!] {w}")
    else:
        print(" [OK] Văn bản đang có hiệu lực bình thường, không có cảnh báo.")
    print("=======================================================\n")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Legal RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ask
    ask_parser = subparsers.add_parser("ask", help="Ask a legal question")
    ask_parser.add_argument("question", type=str, help="Question text")
    ask_parser.add_argument("--fake", action="store_true", help="Run with fake models and data")

    # Retrieve
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve legal contexts for a query")
    retrieve_parser.add_argument("query", type=str, help="Search query")
    retrieve_parser.add_argument("--top-k", type=int, default=5, help="Number of parent contexts to retrieve")
    retrieve_parser.add_argument("--scope", choices=["corpus", "upload", "both"], default="corpus", help="Retrieval scope")
    retrieve_parser.add_argument("--fake", action="store_true", help="Use fake retriever")

    # Check validity
    validity_parser = subparsers.add_parser("check-validity", help="Check document validity and warnings")
    validity_parser.add_argument("doc_number", type=str, help="Document number or doc_id (e.g. 10/2012/QH13)")

    args = parser.parse_args()

    setup_logging(level="WARNING")

    if args.command == "ask":
        if args.fake:
            ask_fake(args.question)
        else:
            ask_real(args.question)
    elif args.command == "retrieve":
        run_retrieval(args.query, top_k=args.top_k, scope=args.scope, fake=args.fake)
    elif args.command == "check-validity":
        run_check_validity(args.doc_number)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

