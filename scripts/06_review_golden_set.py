"""Script 06: Interactive CLI and CSV exporter for reviewing Golden Set items."""

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

from legalrag.evaluation.datasets import load_golden_set, save_golden_set


def print_summary(items: list) -> None:
    print("\n--- Golden Set Review Summary ---")
    df = pd.DataFrame([it.model_dump() for it in items])
    if df.empty:
        print("Golden set is empty.")
        return

    print(f"Total Questions: {len(df)}")
    print("\nBy Category:")
    print(df["category"].value_counts().to_string())

    print("\nBy Review Status:")
    print(df["review_status"].value_counts().to_string())

    print("\nBy Split:")
    print(df["split"].value_counts().to_string())


def export_csv(items: list, csv_path: Path) -> None:
    csv_rows = []
    for it in items:
        gold_arts_str = "; ".join(f"{g.doc_id}:A{g.article_no}" for g in it.gold_articles)
        csv_rows.append({
            "id": it.id,
            "category": it.category,
            "split": it.split,
            "question": it.question,
            "gold_articles": gold_arts_str,
            "reference_answer": it.reference_answer,
            "expected_behavior": it.expected_behavior,
            "expected_validity": it.expected_validity,
            "review_status": it.review_status,
            "reviewer_note": it.reviewer_note,
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Exported {len(csv_df)} items to CSV at {csv_path}")


def interactive_review(items: list, file_path: Path) -> None:
    pending_items = [it for it in items if it.review_status == "pending"]
    print(f"Found {len(pending_items)} pending items to review (out of {len(items)} total).")

    if not pending_items:
        print("All items have already been reviewed!")
        print_summary(items)
        return

    print("Keys: [a]ccept, [r]eject, [e]dit, [s]kip, [q]uit")

    for i, it in enumerate(pending_items, start=1):
        print(f"\n[{i}/{len(pending_items)}] ID: {it.id} | Category: {it.category} | Split: {it.split}")
        print(f"Question : {it.question}")
        print(f"Gold Arts: {', '.join(f'{g.doc_id}:A{g.article_no}' for g in it.gold_articles)}")
        print(f"Ref Ans  : {it.reference_answer[:200]}...")

        choice = input("Choice (a/r/e/s/q) [a]: ").strip().lower() or "a"
        if choice == "a":
            it.review_status = "accepted"
        elif choice == "r":
            it.review_status = "rejected"
        elif choice == "e":
            new_q = input(f"New question [{it.question}]: ").strip()
            if new_q:
                it.question = new_q
            it.review_status = "edited"
        elif choice == "s":
            continue
        elif choice == "q":
            print("Exiting review.")
            break

        # Save immediately for progress persistence
        save_golden_set(items, file_path)

    print_summary(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Golden Set items")
    parser.add_argument("--file", default="eval/golden_set.jsonl", help="Path to golden_set.jsonl")
    parser.add_argument("--export-csv", default="eval/golden_set.csv", help="Path to export CSV")
    parser.add_argument("--status-only", action="store_true", help="Print summary status without interactive mode")
    parser.add_argument("--accept-all", action="store_true", help="Accept all pending questions")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        sys.exit(1)

    items = load_golden_set(file_path)

    if args.accept_all:
        for it in items:
            if it.review_status == "pending":
                it.review_status = "accepted"
        save_golden_set(items, file_path)
        print("All pending items have been accepted.")

    if args.export_csv:
        export_csv(items, Path(args.export_csv))

    if args.status_only or args.accept_all:
        print_summary(items)
    else:
        interactive_review(items, file_path)


if __name__ == "__main__":
    main()
