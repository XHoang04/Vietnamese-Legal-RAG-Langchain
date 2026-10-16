"""Chunk documents from corpus_docs.parquet and generate parents.parquet, chunks.parquet, stats, and samples."""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from legalrag.core.config import load_config
from legalrag.data.db import get_engine, parents_table
from legalrag.data.models import LegalDocument
from legalrag.ingestion.legal_chunker import chunk_document


def run_chunking() -> None:
    config_bundle = load_config()
    chunking_cfg = config_bundle.app.chunking
    split_clauses = config_bundle.retrieval.parent_child.enabled

    corpus_path = Path("data/interim/corpus_docs.parquet")
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found. Please run scripts/01_download_filter.py first.")
        sys.exit(1)

    print(f"--- Building Chunks from {corpus_path} ---")
    df = pd.read_parquet(corpus_path)
    total_docs = len(df)
    print(f"Loaded {total_docs} documents")

    all_parents = []
    all_chunks = []
    level_counts = Counter()
    chunk_tokens_list = []
    has_struct_count = 0

    doc_samples_info = []

    for _, row in tqdm(df.iterrows(), desc="Chunking documents", total=total_docs):
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        doc = LegalDocument(**row_dict)
        parents, chunks = chunk_document(
            doc=doc,
            cfg=chunking_cfg,
            split_clauses=split_clauses,
        )

        all_parents.extend(parents)
        all_chunks.extend(chunks)

        is_struct = any(p.article_no is not None for p in parents)
        if is_struct:
            has_struct_count += 1

        for c in chunks:
            level_counts[c.level] += 1
            chunk_tokens_list.append(c.token_count)

        # Collect sample for review
        if len(doc_samples_info) < 20:
            doc_samples_info.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "num_parents": len(parents),
                "num_chunks": len(chunks),
                "first_3_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "breadcrumb": c.breadcrumb,
                        "level": c.level,
                        "tokens": c.token_count,
                        "text_sample": c.text[:200],
                    }
                    for c in chunks[:3]
                ],
            })

    # Prepare outputs
    processed_dir = Path("data/processed")
    interim_dir = Path("data/interim")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save chunks.parquet
    chunks_records = [c.model_dump() for c in all_chunks]
    pd.DataFrame(chunks_records).to_parquet(processed_dir / "chunks.parquet", index=False)
    print(f"Saved {len(all_chunks)} chunks to {processed_dir / 'chunks.parquet'}")

    # 2. Save parents.parquet
    parents_records = [p.model_dump() for p in all_parents]
    pd.DataFrame(parents_records).to_parquet(processed_dir / "parents.parquet", index=False)
    print(f"Saved {len(all_parents)} parents to {processed_dir / 'parents.parquet'}")

    # 3. Calculate statistics
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": total_docs,
        "total_parents": len(all_parents),
        "total_chunks": len(all_chunks),
        "documents_with_structure_count": has_struct_count,
        "documents_with_structure_ratio": has_struct_count / total_docs if total_docs else 0.0,
        "chunks_by_level": dict(level_counts),
        "token_percentiles": {
            "p50": float(np.percentile(chunk_tokens_list, 50)) if chunk_tokens_list else 0.0,
            "p90": float(np.percentile(chunk_tokens_list, 90)) if chunk_tokens_list else 0.0,
            "p99": float(np.percentile(chunk_tokens_list, 99)) if chunk_tokens_list else 0.0,
        },
    }

    with open(interim_dir / "chunk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 4. Generate visual review samples: chunk_samples.md
    sample_md_lines = [
        "# Mẫu Đọc Kiểm Tra Chunking (chunk_samples.md)",
        "",
        f"- Tổng số văn bản trong tập con: {total_docs}",
        f"- Tổng số Parent: {len(all_parents)}",
        f"- Tổng số Chunk: {len(all_chunks)}",
        f"- Tỉ lệ văn bản có cấu trúc Điều: {stats['documents_with_structure_ratio']*100:.1f}%",
        f"- Phân bố chunk theo cấp: {dict(level_counts)}",
        "",
        "---",
    ]

    for s in doc_samples_info:
        sample_md_lines.append(f"\n## Văn bản: {s['title']} (ID: {s['doc_id']})")
        sample_md_lines.append(f"- Số Parent: {s['num_parents']}, Số Chunk: {s['num_chunks']}")
        sample_md_lines.append("\n### 3 Chunk đầu tiên:")
        for idx, c in enumerate(s["first_3_chunks"]):
            sample_md_lines.append(f"**Chunk {idx+1}** [`{c['chunk_id']}`] (Cấp: `{c['level']}`, Tokens: `{c['tokens']}`)")
            sample_md_lines.append(f"> **Breadcrumb**: {c['breadcrumb']}")
            sample_md_lines.append(f"```text\n{c['text_sample']}...\n```")

    with open(interim_dir / "chunk_samples.md", "w", encoding="utf-8") as f:
        f.write("\n".join(sample_md_lines))

    # 5. Save parents to SQL database
    engine = get_engine(config_bundle.app.db.url)
    sql_parents = [
        {
            "parent_id": p.parent_id,
            "doc_id": p.doc_id,
            "article_no": p.article_no,
            "article_title": p.article_title,
            "breadcrumb": p.breadcrumb,
            "text": p.text,
            "token_count": p.token_count,
            "source": p.source,
            "session_id": p.session_id,
            "expires_at": None,
        }
        for p in all_parents
    ]

    with engine.begin() as conn:
        conn.execute(parents_table.delete())
        if sql_parents:
            conn.execute(parents_table.insert(), sql_parents)

    print("Saved parents into SQLite database table 'parents'")
    print(f"Stats: {stats['chunks_by_level']}")
    print("--- Chunking Completed Successfully ---")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    run_chunking()


if __name__ == "__main__":
    main()
