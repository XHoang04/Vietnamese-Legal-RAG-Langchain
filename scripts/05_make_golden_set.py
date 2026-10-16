"""Script 05: Generate Golden Set candidates across 5 categories from corpus."""

import argparse
import random
import re
import sys
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from legalrag.evaluation.datasets import GoldenArticle, GoldenQuestion, save_golden_set


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


def build_single_questions(parents_df: pd.DataFrame, docs_df: pd.DataFrame, count: int = 45) -> list[GoldenQuestion]:
    """Generate self-contained single-hop legal questions from representative articles."""
    questions: list[GoldenQuestion] = []

    # Filter articles with meaningful content
    sub = parents_df[
        parents_df["article_no"].notna()
        & (parents_df["token_count"] >= 50)
        & (parents_df["token_count"] <= 800)
        & parents_df["article_title"].notna()
        & (parents_df["article_title"].str.len() > 5)
    ].copy()

    # Merge doc_number and title
    merged = sub.merge(docs_df[["doc_id", "doc_number", "title"]], on="doc_id")

    # Stratified sample across documents
    sample_df = merged.sample(n=min(count, len(merged)), random_state=42)

    for idx, row in enumerate(sample_df.iterrows(), start=1):
        _, r = row
        doc_id = str(r["doc_id"])
        art_no = str(r["article_no"])
        art_title = str(r["article_title"])
        doc_title = str(r["title"])
        doc_num = str(r.get("doc_number") or doc_title)
        text = str(r["text"])

        # Create natural questions based on article title and topic
        if "quyền" in art_title.lower():
            q_text = f"Theo quy định tại {doc_num}, {art_title.lower()} được quy định cụ thể như thế nào?"
        elif "trách nhiệm" in art_title.lower() or "nghĩa vụ" in art_title.lower():
            q_text = f"Trách nhiệm thực hiện theo {doc_num} về {art_title.lower()} bao gồm những nội dung gì?"
        elif "điều kiện" in art_title.lower() or "thủ tục" in art_title.lower():
            q_text = f"Điều kiện và trình tự thủ tục về {art_title.lower()} theo {doc_num} gồm những gì?"
        elif "thời giờ" in art_title.lower() or "thời gian" in art_title.lower() or "thời hạn" in art_title.lower():
            q_text = f"Quy định về {art_title.lower()} tại {doc_num} quy định thời gian cụ thể là bao lâu?"
        elif "nguyên tắc" in art_title.lower():
            q_text = f"Các nguyên tắc cơ bản về {art_title.lower()} được quy định như thế nào trong {doc_num}?"
        elif "mức" in art_title.lower() or "tiền" in art_title.lower() or "chi phí" in art_title.lower():
            q_text = f"Mức chi trả và quy định về {art_title.lower()} được xác định thế nào theo {doc_num}?"
        else:
            q_text = f"Quy định pháp luật về {art_title.lower()} theo Điều {art_no} của {doc_num} là gì?"

        # Generate reference answer snippet from first 250 characters
        ref_ans = f"Theo Điều {art_no} {doc_num}: {text[:250].strip()}..."

        questions.append(
            GoldenQuestion(
                id=f"single_{idx:03d}",
                category="single",
                question=q_text,
                gold_articles=[GoldenArticle(doc_id=doc_id, article_no=art_no)],
                reference_answer=ref_ans,
                expected_behavior="answer",
                expected_validity="in_force",
                source_parent_ids=[str(r["parent_id"])],
            )
        )

    return questions


def build_multi_step_questions(parents_df: pd.DataFrame, docs_df: pd.DataFrame, count: int = 20) -> list[GoldenQuestion]:
    """Generate multi-step comparative or synthesis questions combining two articles."""
    questions: list[GoldenQuestion] = []

    merged = parents_df[parents_df["article_no"].notna()].merge(
        docs_df[["doc_id", "doc_number", "title"]], on="doc_id"
    )

    # Find documents with multiple articles
    doc_counts = merged["doc_id"].value_counts()
    multi_art_docs = doc_counts[doc_counts >= 4].index.tolist()

    random.seed(42)
    selected_docs = random.sample(multi_art_docs, min(count, len(multi_art_docs)))

    for idx, did in enumerate(selected_docs, start=1):
        doc_arts = merged[merged["doc_id"] == did].sample(n=2, random_state=idx)
        r1, r2 = doc_arts.iloc[0], doc_arts.iloc[1]

        doc_num = str(r1.get("doc_number") or r1["title"])
        art1_no, art1_title = str(r1["article_no"]), str(r1["article_title"])
        art2_no, art2_title = str(r2["article_no"]), str(r2["article_title"])

        q_text = (
            f"Hãy so sánh hoặc tổng hợp quy định giữa Điều {art1_no} ({art1_title}) "
            f"và Điều {art2_no} ({art2_title}) theo văn bản {doc_num}."
        )

        ref_ans = (
            f"Văn bản {doc_num} quy định tại Điều {art1_no} về {art1_title}: {r1['text'][:150]}... "
            f"Đồng thời tại Điều {art2_no} về {art2_title}: {r2['text'][:150]}..."
        )

        questions.append(
            GoldenQuestion(
                id=f"multi_{idx:03d}",
                category="multi_step",
                question=q_text,
                gold_articles=[
                    GoldenArticle(doc_id=str(r1["doc_id"]), article_no=art1_no),
                    GoldenArticle(doc_id=str(r2["doc_id"]), article_no=art2_no),
                ],
                reference_answer=ref_ans,
                expected_behavior="answer",
                expected_validity="in_force",
                source_parent_ids=[str(r1["parent_id"]), str(r2["parent_id"])],
            )
        )

    return questions


def build_validity_questions(docs_df: pd.DataFrame, count: int = 20) -> list[GoldenQuestion]:
    """Generate validity questions assessing in-force, expired, or partially expired status."""
    questions: list[GoldenQuestion] = []

    # Sample expired and in-force documents
    expired_docs = docs_df[docs_df["status"].isin(["EXPIRED", "PARTIALLY_EXPIRED"])].sample(
        n=min(12, len(docs_df[docs_df["status"].isin(["EXPIRED", "PARTIALLY_EXPIRED"])])),
        random_state=42,
    )
    in_force_docs = docs_df[docs_df["status"] == "IN_FORCE"].sample(
        n=min(8, len(docs_df[docs_df["status"] == "IN_FORCE"])),
        random_state=42,
    )

    combined = pd.concat([expired_docs, in_force_docs])
    for idx, (_, r) in enumerate(combined.iterrows(), start=1):
        doc_id = str(r["doc_id"])
        doc_num = str(r.get("doc_number") or r["title"])
        title = str(r["title"])
        status = str(r["status"])

        if status == "EXPIRED":
            exp_val = "expired"
            q_text = f"Văn bản {doc_num} ({title}) hiện nay còn hiệu lực pháp luật hay đã hết hiệu lực?"
            ref_ans = f"Văn bản {title} ({doc_num}) hiện nay đã hết hiệu lực toàn bộ theo dữ liệu cơ sở dữ liệu pháp luật."
        elif status == "PARTIALLY_EXPIRED":
            exp_val = "partial"
            q_text = f"Văn bản {doc_num} ({title}) hiện còn hiệu lực toàn bộ hay đã hết hiệu lực một phần?"
            ref_ans = f"Văn bản {title} ({doc_num}) đã hết hiệu lực một phần, cần kiểm tra đối chiếu văn bản sửa đổi bổ sung."
        else:
            exp_val = "in_force"
            q_text = f"Văn bản {doc_num} ({title}) có còn hiệu lực thi hành không?"
            ref_ans = f"Văn bản {title} ({doc_num}) hiện nay đang còn hiệu lực thi hành."

        questions.append(
            GoldenQuestion(
                id=f"val_{idx:03d}",
                category="validity",
                question=q_text,
                gold_articles=[GoldenArticle(doc_id=doc_id, article_no="1")],
                reference_answer=ref_ans,
                expected_behavior="answer",
                expected_validity=exp_val,
                source_parent_ids=[f"{doc_id}:A1"],
            )
        )

    return questions


def build_out_of_scope_questions(count: int = 14) -> list[GoldenQuestion]:
    """Generate out-of-scope non-legal or other-domain questions with expected refusal."""
    prompts = [
        "Cách làm món phở bò truyền thống tại nhà thơm ngon nhất?",
        "Viết một hàm Python để sắp xếp mảng theo giải thuật quicksort.",
        "Quy định về mức xử phạt vi phạm nồng độ cồn khi lái xe ô tô theo luật giao thông đường bộ?",
        "Mức hình phạt đối với tội trộm cắp tài sản theo Bộ luật Hình sự là bao nhiêu năm tù?",
        "Thủ tục đăng ký kết hôn có yếu tố nước ngoài tại Sở Tư pháp?",
        "Khoảng cách từ Trái Đất đến Mặt Trăng là bao nhiêu km?",
        "Thời tiết Hà Nội ngày mai dự báo có mưa hay nắng?",
        "Cách sửa lỗi kết nối wifi bị chấm than vàng trên hệ điều hành Windows?",
        "Chỉ số VN-Index hôm nay tăng hay giảm bao nhiêu điểm?",
        "Quy trình cấp sổ đỏ và chuyển nhượng quyền sử dụng đất đai cho cá nhân?",
        "Bảo hiểm trách nhiệm dân sự bắt buộc của chủ xe mô tô xe máy mua ở đâu?",
        "Thủ tục xin cấp hộ chiếu phổ thông trực tuyến trên Cổng dịch vụ công quốc gia?",
        "Đội bóng nào đã giành chức vô địch giải bóng đá Ngoại hạng Anh mùa giải trước?",
        "Hướng dẫn cách chăm sóc cây phong lan ra hoa đúng dịp Tết Nguyên Đán?",
    ]

    questions: list[GoldenQuestion] = []
    for idx, q_text in enumerate(prompts[:count], start=1):
        questions.append(
            GoldenQuestion(
                id=f"oos_{idx:03d}",
                category="out_of_scope",
                question=q_text,
                gold_articles=[],
                reference_answer="Câu hỏi nằm ngoài phạm vi tư vấn về pháp luật lao động Việt Nam của hệ thống.",
                expected_behavior="refuse",
                expected_validity="none",
                source_parent_ids=[],
            )
        )
    return questions


def build_followup_questions(parents_df: pd.DataFrame, docs_df: pd.DataFrame, count: int = 10) -> list[GoldenQuestion]:
    """Generate multi-turn conversational followup questions with pronouns."""
    questions: list[GoldenQuestion] = []

    sub = parents_df[
        parents_df["article_no"].notna()
        & parents_df["article_title"].notna()
        & (parents_df["token_count"] >= 80)
    ].merge(docs_df[["doc_id", "doc_number", "title"]], on="doc_id")

    sample_sub = sub.sample(n=min(count, len(sub)), random_state=84)

    for idx, (_, r) in enumerate(sample_sub.iterrows(), start=1):
        doc_id = str(r["doc_id"])
        art_no = str(r["article_no"])
        art_title = str(r["article_title"])
        doc_num = str(r.get("doc_number") or r["title"])
        text = str(r["text"])

        initial_q = f"Cho tôi hỏi về quy định {art_title} tại {doc_num}?"
        initial_a = f"Điều {art_no} của {doc_num} quy định chi tiết về {art_title} đối với các bên liên quan."
        followup_q = "Vậy trong trường hợp này, quyền và trách nhiệm cụ thể được áp dụng ra sao?"

        questions.append(
            GoldenQuestion(
                id=f"follow_{idx:03d}",
                category="followup",
                question=followup_q,
                history=[
                    {"role": "user", "content": initial_q},
                    {"role": "assistant", "content": initial_a},
                ],
                gold_articles=[GoldenArticle(doc_id=doc_id, article_no=art_no)],
                reference_answer=f"Căn cứ Điều {art_no} {doc_num}: {text[:200]}...",
                expected_behavior="answer",
                expected_validity="in_force",
                source_parent_ids=[str(r["parent_id"])],
            )
        )

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Golden Set for Legal RAG evaluation")
    parser.add_argument("--parents", default="data/processed/parents.parquet", help="Path to parents.parquet")
    parser.add_argument("--docs", default="data/interim/corpus_docs.parquet", help="Path to corpus_docs.parquet")
    parser.add_argument("--out", default="eval/golden_set.jsonl", help="Output JSON Lines file")
    parser.add_argument("--dev-ratio", type=float, default=0.3, help="Proportion allocated to dev split (default 0.3)")
    parser.add_argument("--auto-accept", action="store_true", help="Set review_status='accepted' automatically")
    args = parser.parse_args()

    parents_path = Path(args.parents)
    docs_path = Path(args.docs)
    out_path = Path(args.out)

    if not parents_path.exists() or not docs_path.exists():
        print(f"Error: Required files not found: {parents_path} or {docs_path}")
        sys.exit(1)

    print("--- Generating Golden Set Candidates ---")
    parents_df = pd.read_parquet(parents_path)
    docs_df = pd.read_parquet(docs_path)

    # 1. Single (45)
    singles = build_single_questions(parents_df, docs_df, count=45)
    print(f"Generated {len(singles)} single-hop questions.")

    # 2. Multi-step (20)
    multis = build_multi_step_questions(parents_df, docs_df, count=20)
    print(f"Generated {len(multis)} multi-step questions.")

    # 3. Validity (20)
    vals = build_validity_questions(docs_df, count=20)
    print(f"Generated {len(vals)} validity questions.")

    # 4. Out of scope (14)
    ooss = build_out_of_scope_questions(count=14)
    print(f"Generated {len(ooss)} out-of-scope questions.")

    # 5. Followup (10)
    follows = build_followup_questions(parents_df, docs_df, count=10)
    print(f"Generated {len(follows)} follow-up questions.")

    all_items = singles + multis + vals + ooss + follows
    total = len(all_items)
    print(f"Total candidate questions: {total}")

    # Assign splits: 30% dev, 70% test stratified per category
    random.seed(42)
    categories = ["single", "multi_step", "validity", "out_of_scope", "followup"]
    for cat in categories:
        cat_items = [it for it in all_items if it.category == cat]
        random.shuffle(cat_items)
        dev_count = int(len(cat_items) * args.dev_ratio)
        for i, it in enumerate(cat_items):
            it.split = "dev" if i < dev_count else "test"
            if args.auto_accept:
                it.review_status = "accepted"

    save_golden_set(all_items, out_path)
    print(f"\nSaved {total} questions to {out_path}")

    # Breakdown
    dev_n = sum(1 for q in all_items if q.split == "dev")
    test_n = sum(1 for q in all_items if q.split == "test")
    accepted_n = sum(1 for q in all_items if q.review_status == "accepted")
    print(f"Split breakdown: dev = {dev_n} ({dev_n/total*100:.1f}%), test = {test_n} ({test_n/total*100:.1f}%)")
    print(f"Review status: accepted = {accepted_n}, pending = {total - accepted_n}")


if __name__ == "__main__":
    main()
