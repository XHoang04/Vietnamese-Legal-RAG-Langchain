"""Unit tests for syllable-based BM25 sparse tokenizer and encoder."""

from legalrag.retrieval.sparse import (
    SyllableBM25,
    compute_idf,
    hash_token,
    tokenize_syllables,
)


def test_tokenize_syllables_basic():
    text = "Thời giờ làm việc, thời giờ nghỉ ngơi."
    tokens = tokenize_syllables(text, mode="syllable")
    assert tokens == ["thời", "giờ", "làm", "việc", "thời", "giờ", "nghỉ", "ngơi"]


def test_tokenize_syllables_bigram():
    text = "thời giờ làm việc"
    tokens = tokenize_syllables(text, mode="syllable_bigram")
    # 4 unigrams + 3 bigrams
    assert tokens == [
        "thời", "giờ", "làm", "việc",
        "thời_giờ", "giờ_làm", "làm_việc",
    ]


def test_hash_token_deterministic():
    token = "thời_giờ"
    h1 = hash_token(token)
    h2 = hash_token(token)
    assert h1 == h2
    assert isinstance(h1, int)
    assert 0 <= h1 < (2 ** 32)


def test_sparse_bm25_relevance_ranking():
    corpus = [
        "Quy định về thời giờ làm việc và thời giờ nghỉ ngơi của người lao động.",
        "Tiền lương tối thiểu vùng theo tháng và theo giờ áp dụng đối với người lao động.",
        "Kỷ luật lao động và trách nhiệm vật chất trong doanh nghiệp sản xuất.",
    ]
    idf_table, avgdl = compute_idf(corpus, mode="syllable_bigram")
    bm25 = SyllableBM25(mode="syllable_bigram", avgdl=avgdl, idf_table=idf_table)

    query_vec = bm25.encode_query("thời giờ làm việc")
    doc_vecs = [bm25.encode_document(doc) for doc in corpus]

    def dot_product(v1, v2) -> float:
        d1 = v1.as_dict()
        d2 = v2.as_dict()
        common_keys = set(d1.keys()) & set(d2.keys())
        return sum(d1[k] * d2[k] for k in common_keys)

    scores = [dot_product(query_vec, dv) for dv in doc_vecs]
    # Document 0 must have significantly higher score than doc 1 and doc 2
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
