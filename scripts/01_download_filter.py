"""Download, filter, normalize, and extract domain subset from ThucPD/vbpl-vn."""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from legalrag.core.config import load_config
from legalrag.data.db import get_engine, init_db, save_documents_and_relations
from legalrag.data.filters import evaluate_document_filter
from legalrag.data.hf_loader import stream_vbpl_dataset
from legalrag.data.metadata import (
    extract_subject,
    parse_date,
    parse_doc_number,
    parse_status,
)
from legalrag.data.models import LegalDocument, Relation
from legalrag.data.relations import extract_relations_from_schema
from legalrag.data.subset import select_domain_subset


def run_pipeline(limit: int | None = None) -> None:
    config_bundle = load_config()
    domain_cfg = config_bundle.domain
    filters_cfg = config_bundle.app.filters

    print(f"--- Starting Data Ingestion Pipeline (domain={domain_cfg.domain}, limit={limit}) ---")

    filtered_out_records = []
    filter_counts = Counter()
    documents_raw: dict[str, LegalDocument] = {}
    relations_raw: list[Relation] = []

    stream = stream_vbpl_dataset(limit=limit)

    for row in tqdm(stream, desc="Processing raw records", total=limit):
        source_id = row["source_id"]
        url = row["url"]
        title = row["title"]
        last_updated_str = row["last_updated_time"]
        full_text = row["full_text"]
        attribute = row["attribute"]
        schema = row["schema"]

        if not source_id or not title:
            continue

        status, status_raw = parse_status(attribute)
        subject = extract_subject(attribute)

        # Evaluate filters
        filter_reason, metadata_flag, final_status = evaluate_document_filter(
            url=url,
            full_text=full_text,
            subject=subject,
            status=status,
            status_raw=status_raw,
            cfg=filters_cfg,
        )

        if filter_reason:
            filter_counts[filter_reason] += 1
            filtered_out_records.append({
                "source_id": source_id,
                "title": title,
                "reason": filter_reason,
            })
            continue

        doc_type, doc_number, doc_number_norm = parse_doc_number(
            title, allowed_doc_types=domain_cfg.allowed_doc_types
        )

        # Parse date fields
        eff_date = parse_date(attribute.get("effective_date"))
        enf_date = parse_date(attribute.get("enforced_date"))
        exp_date = parse_date(attribute.get("expiry_date"))
        gaz_date = parse_date(attribute.get("gazette_date"))

        last_updated_dt = None
        if last_updated_str:
            try:
                last_updated_dt = datetime.fromisoformat(last_updated_str.replace("Z", ""))
            except ValueError:
                pass

        eff_area = attribute.get("effective_area")
        eff_area_str = str(eff_area).strip() if eff_area else None

        doc = LegalDocument(
            doc_id=source_id,
            doc_number=doc_number,
            doc_number_norm=doc_number_norm,
            title=title,
            subject=subject,
            doc_type=doc_type,
            effective_area=eff_area_str,
            status=final_status,
            status_raw=status_raw,
            effective_date=eff_date,
            enforced_date=enf_date,
            expiry_date=exp_date,
            gazette_date=gaz_date,
            url=url,
            last_updated_time=last_updated_dt,
            full_text=full_text,
            metadata_flag=metadata_flag,
            source="corpus",
        )

        # Handle duplicates: keep latest last_updated_time
        if source_id in documents_raw:
            existing = documents_raw[source_id]
            if doc.last_updated_time and existing.last_updated_time:
                if doc.last_updated_time > existing.last_updated_time:
                    documents_raw[source_id] = doc
                    filter_counts["DUPLICATE"] += 1
            else:
                filter_counts["DUPLICATE"] += 1
        else:
            documents_raw[source_id] = doc

        # Extract relations from schema
        doc_relations = extract_relations_from_schema(source_id, schema)
        relations_raw.extend(doc_relations)

    all_docs = list(documents_raw.values())
    print(f"Total valid candidate documents after base filters: {len(all_docs)}")

    # Select domain subset
    final_docs, final_relations, subset_meta = select_domain_subset(
        all_docs, relations_raw, domain_cfg
    )

    print(f"Selected {len(final_docs)} documents for domain '{domain_cfg.domain}'")
    print(f"Selected {len(final_relations)} associated relations")

    # Output directory
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save corpus_docs.parquet
    docs_records = [d.model_dump() for d in final_docs]
    docs_df = pd.DataFrame(docs_records)
    docs_df.to_parquet(interim_dir / "corpus_docs.parquet", index=False)

    # 2. Save relations.parquet
    rels_records = [r.model_dump() for r in final_relations]
    rels_df = pd.DataFrame(rels_records)
    rels_df.to_parquet(interim_dir / "relations.parquet", index=False)

    # 3. Save filter_report.json
    filter_report = {
        "total_filtered_out": len(filtered_out_records),
        "counts_by_reason": dict(filter_counts),
    }
    with open(interim_dir / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(filter_report, f, ensure_ascii=False, indent=2)

    # 4. Save filtered_out.csv
    with open(interim_dir / "filtered_out.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "title", "reason"])
        writer.writeheader()
        writer.writerows(filtered_out_records)

    # 5. Save subset_report.md
    subset_md_lines = [
        f"# Báo cáo Tập con Lĩnh vực ({domain_cfg.domain})",
        "",
        f"- Tổng số văn bản: {len(final_docs)}",
        f"- Tổng số quan hệ: {len(final_relations)}",
        f"- Số hiệu hạt giống thiếu (không tìm thấy): {subset_meta.get('missing_seeds', [])}",
        "",
        "## Phân bố loại văn bản:",
    ]
    type_counts = Counter(d.doc_type or "Chưa rõ" for d in final_docs)
    for dt, cnt in type_counts.most_common():
        subset_md_lines.append(f"- **{dt}**: {cnt}")

    status_counts = Counter(d.status.value for d in final_docs)
    subset_md_lines.append("\n## Phân bố trạng thái hiệu lực:")
    for st, cnt in status_counts.most_common():
        subset_md_lines.append(f"- **{st}**: {cnt}")

    with open(interim_dir / "subset_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(subset_md_lines))

    # 6. Save to SQL database
    db_url = config_bundle.app.db.url
    print(f"Saving to database: {db_url}")
    engine = get_engine(db_url)
    init_db(engine)

    # Prepare SQL dicts
    sql_docs = []
    for d in final_docs:
        sql_docs.append({
            "doc_id": d.doc_id,
            "doc_number": d.doc_number,
            "doc_number_norm": d.doc_number_norm,
            "title": d.title,
            "subject": d.subject,
            "doc_type": d.doc_type,
            "effective_area": d.effective_area,
            "status": d.status.value,
            "status_raw": d.status_raw,
            "effective_date": d.effective_date,
            "enforced_date": d.enforced_date,
            "expiry_date": d.expiry_date,
            "gazette_date": d.gazette_date,
            "url": d.url,
            "last_updated_time": d.last_updated_time,
            "metadata_flag": d.metadata_flag,
        })

    sql_rels = []
    for r in final_relations:
        sql_rels.append({
            "src_doc_id": r.src_doc_id,
            "rel_type": r.rel_type.value,
            "dst_doc_id": r.dst_doc_id,
            "dst_title": r.dst_title,
            "dst_url": r.dst_url,
            "raw_key": r.raw_key,
            "resolved": r.resolved,
        })

    save_documents_and_relations(engine, sql_docs, sql_rels)
    print("--- Ingestion Pipeline Completed Successfully ---")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Download and filter dataset")
    parser.add_argument("--limit", type=int, default=15000, help="Number of rows to stream (default 15000)")
    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit
    run_pipeline(limit=limit)


if __name__ == "__main__":
    main()
