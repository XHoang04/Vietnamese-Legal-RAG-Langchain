"""Inspect Hugging Face dataset ThucPD/vbpl-vn and generate inspect_report.json."""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def extract_item_id(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"ItemID=(\d+)", url)
    return match.group(1) if match else None


def tokenize_for_overlap(text: str) -> set[str]:
    text = text.lower()
    tokens = re.findall(r"\w+", text, re.UNICODE)
    return set(tokens)


def inspect_dataset(limit: int | None = None, out_file: str = "data/interim/inspect_report.json") -> None:
    print(f"Loading dataset ThucPD/vbpl-vn (streaming=True)... (limit={limit})")
    ds = load_dataset("ThucPD/vbpl-vn", split="train", streaming=True)

    # Metrics containers
    total_rows = 0
    samples = []
    url_patterns = Counter()
    empty_or_short_text = 0
    text_lengths = []
    attribute_keys_counter = Counter()
    schema_keys_counter = Counter()
    status_raw_counter = Counter()
    date_regex = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    valid_dates_count = 0
    total_dates_checked = 0
    doc_types_counter = Counter()
    effective_areas_counter = Counter()
    has_article_struct_count = 0
    newline_densities = []
    mismatch_flags_count = 0

    all_source_ids = set()
    relations_forward = {
        "amend": [],
        "cancel": [],
        "cancel_one_part": [],
    }
    expired_total_count = 0
    expired_with_cancel_count = 0

    article_regex = re.compile(r"^Điều\s+(\d+[a-zA-Z]?)\s*[\.:]", re.MULTILINE)

    for row in tqdm(ds, desc="Inspecting rows", total=limit):
        total_rows += 1
        source_id = str(row.get("source_id", "")).strip()
        url = str(row.get("url", "")).strip()
        title = str(row.get("title", "")).strip()
        last_updated = str(row.get("last_updated_time", "")).strip()
        full_text = str(row.get("full_text", "") or "")
        attribute = row.get("attribute") or {}
        schema = row.get("schema") or {}

        if source_id:
            all_source_ids.add(source_id)

        # 1. First 5 samples (cut text to 300 chars)
        if len(samples) < 5:
            samples.append({
                "source_id": source_id,
                "url": url,
                "title": title,
                "last_updated_time": last_updated,
                "full_text_sample": full_text[:300],
                "attribute_sample": attribute,
                "schema_keys": list(schema.keys()),
            })

        # 2. URL distribution
        if "vbpq-toanvan.aspx" in url:
            url_patterns["vbpq-toanvan"] += 1
        elif "vbpqen-toanvan.aspx" in url:
            url_patterns["vbpqen-toanvan"] += 1
        elif "vbpq-van-ban-goc.aspx" in url:
            url_patterns["vbpq-van-ban-goc"] += 1
        else:
            url_patterns["other"] += 1

        # 3. Text length distribution
        text_len = len(full_text.strip())
        text_lengths.append(text_len)
        if text_len < 200:
            empty_or_short_text += 1

        # 4. Attribute and Schema keys
        if isinstance(attribute, dict):
            for k in attribute.keys():
                attribute_keys_counter[k] += 1
        if isinstance(schema, dict):
            for k in schema.keys():
                schema_keys_counter[k] += 1

        # 5. Status raw, dates, doc types, effective area
        doc_info = attribute.get("document_info") or []
        subject = None
        status_raw = None
        if isinstance(doc_info, list):
            for item in doc_info:
                item_str = str(item).strip()
                if item_str.lower().startswith("tình trạng:"):
                    status_raw = item_str.split(":", 1)[1].strip()
                    status_raw_counter[status_raw] += 1
                elif subject is None and item_str:
                    subject = item_str
        elif isinstance(doc_info, str):
            subject = doc_info

        # Check date fields
        for date_key in ["effective_date", "enforced_date", "expiry_date", "gazette_date"]:
            val = attribute.get(date_key)
            if val and isinstance(val, str) and val.strip():
                total_dates_checked += 1
                if date_regex.match(val.strip()):
                    valid_dates_count += 1

        # Doc type
        doc_type_val = attribute.get("document_type")
        if isinstance(doc_type_val, list):
            for dt in doc_type_val:
                doc_types_counter[str(dt).strip()] += 1
        elif isinstance(doc_type_val, str) and doc_type_val.strip():
            doc_types_counter[doc_type_val.strip()] += 1

        # Effective area
        eff_area = attribute.get("effective_area")
        if eff_area and isinstance(eff_area, str) and eff_area.strip():
            effective_areas_counter[eff_area.strip()] += 1

        # 6. Article structure & newline density
        if article_regex.search(full_text):
            has_article_struct_count += 1

        if text_len > 0:
            newlines_count = full_text.count("\n")
            density = (newlines_count / text_len) * 1000
            newline_densities.append(density)

        # 9. Status Hết hiệu lực toàn bộ check cancel_documents
        is_expired = status_raw and "hết hiệu lực toàn bộ" in status_raw.lower()
        cancel_docs = schema.get("cancel_documents") or []
        if is_expired:
            expired_total_count += 1
            if cancel_docs:
                expired_with_cancel_count += 1

        # 10. Mismatch overlap check
        if subject and text_len >= 200:
            subj_tokens = tokenize_for_overlap(subject)
            head_tokens = tokenize_for_overlap(full_text[:600])
            if subj_tokens and head_tokens:
                overlap = len(subj_tokens & head_tokens) / len(subj_tokens)
                if overlap < 0.3:
                    mismatch_flags_count += 1

        # Track relations for reciprocity check
        for rel_name, raw_key in [("amend", "amend_documents"), ("cancel", "cancel_documents"), ("cancel_one_part", "cancel_one_part_documents")]:
            pairs = schema.get(raw_key) or []
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, list) and len(pair) >= 2:
                        dst_url = pair[1]
                        dst_id = extract_item_id(dst_url)
                        if dst_id and source_id:
                            relations_forward[rel_name].append((source_id, dst_id))

        if limit and total_rows >= limit:
            break

    # Quantiles of text length
    p50_len = float(np.percentile(text_lengths, 50)) if text_lengths else 0.0
    p90_len = float(np.percentile(text_lengths, 90)) if text_lengths else 0.0
    p99_len = float(np.percentile(text_lengths, 99)) if text_lengths else 0.0

    # Quantiles of newline density
    median_density = float(np.median(newline_densities)) if newline_densities else 0.0

    # Date regex match ratio
    date_match_ratio = (valid_dates_count / total_dates_checked) if total_dates_checked > 0 else 0.0

    # Expired with cancel ratio
    expired_cancel_ratio = (expired_with_cancel_count / expired_total_count) if expired_total_count > 0 else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_rows_inspected": total_rows,
        "samples_first_5": samples,
        "url_distribution": dict(url_patterns),
        "text_length_stats": {
            "empty_or_short_under_200": empty_or_short_text,
            "empty_or_short_ratio": empty_or_short_text / total_rows if total_rows else 0.0,
            "p50_chars": p50_len,
            "p90_chars": p90_len,
            "p99_chars": p99_len,
        },
        "attribute_keys_frequency": dict(attribute_keys_counter),
        "schema_keys_frequency": dict(schema_keys_counter),
        "status_raw_distribution": dict(status_raw_counter),
        "dates_stats": {
            "total_dates_checked": total_dates_checked,
            "valid_dates_count": valid_dates_count,
            "match_ratio": date_match_ratio,
        },
        "document_types": dict(doc_types_counter),
        "top_20_effective_areas": dict(effective_areas_counter.most_common(20)),
        "structure_stats": {
            "has_article_structure_count": has_article_struct_count,
            "has_article_structure_ratio": has_article_struct_count / total_rows if total_rows else 0.0,
            "median_newline_density_per_1000_chars": median_density,
        },
        "expired_documents_stats": {
            "total_expired_all": expired_total_count,
            "with_cancel_documents": expired_with_cancel_count,
            "ratio_with_cancel_docs": expired_cancel_ratio,
        },
        "metadata_mismatch_stats": {
            "mismatch_count": mismatch_flags_count,
            "mismatch_ratio": mismatch_flags_count / total_rows if total_rows else 0.0,
        },
        "relations_summary": {
            rel_name: len(pairs) for rel_name, pairs in relations_forward.items()
        },
    }

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved inspect report to {out_path}")
    print(f"Total rows inspected: {total_rows}")
    print(f"URL distribution: {dict(url_patterns)}")
    print(f"Has article structure: {has_article_struct_count}/{total_rows} ({report['structure_stats']['has_article_structure_ratio']*100:.1f}%)")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Inspect dataset ThucPD/vbpl-vn")
    parser.add_argument("--limit", type=int, default=5000, help="Number of rows to stream (default 5000 for survey, or 0 for all)")
    parser.add_argument("--out", type=str, default="data/interim/inspect_report.json", help="Output JSON path")
    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit
    inspect_dataset(limit=limit, out_file=args.out)


if __name__ == "__main__":
    main()
