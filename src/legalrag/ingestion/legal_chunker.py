"""Hierarchical legal document chunker for Vietnamese laws, decrees, and circulars."""

import re
from typing import Callable

from legalrag.core.config import ChunkingConfig
from legalrag.core.ids import hash_text, make_chunk_id, make_parent_id
from legalrag.core.text import count_tokens as default_count_tokens
from legalrag.core.text import normalize_whitespace
from legalrag.ingestion.chunk_models import Chunk, ChunkSource, Parent

RE_PART = re.compile(r"^(?:PHẦN|Phần)\s+(?:THỨ\s+\w+|[IVXLC]+|\d+)\b", re.UNICODE)
RE_CHAPTER = re.compile(r"^(?:CHƯƠNG|Chương)\s+([IVXLC]+|\d+)\b", re.UNICODE)
RE_SECTION = re.compile(r"^(?:MỤC|Mục)\s+(\d+|[IVXLC]+)\b", re.UNICODE)
RE_ARTICLE = re.compile(r"^Điều\s*(\d+[a-zA-Z]?)\s*[\.:]\s*(.*)$", re.UNICODE)
RE_CLAUSE = re.compile(r"^(\d{1,2})\s*[\.\)]\s+(.+)$", re.UNICODE)
RE_POINT = re.compile(r"^([a-zđ])\)\s+(.+)$", re.UNICODE)
RE_APPENDIX = re.compile(r"^(?:PHỤ LỤC|Phụ lục)\b", re.UNICODE)


def split_text_into_windows(
    text: str,
    window_tokens: int,
    overlap_tokens: int,
    count_fn: Callable[[str], int],
) -> list[str]:
    """Split long text into sliding windows along paragraph boundaries if possible."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    windows = []
    current_paras: list[str] = []
    current_tokens = 0

    for p in paragraphs:
        p_tok = count_fn(p)
        if current_paras and (current_tokens + p_tok > window_tokens):
            windows.append("\n".join(current_paras))
            # Keep overlap paragraphs from the end
            overlap_paras: list[str] = []
            overlap_tok = 0
            for prev_p in reversed(current_paras):
                prev_tok = count_fn(prev_p)
                if overlap_tok + prev_tok <= overlap_tokens:
                    overlap_paras.insert(0, prev_p)
                    overlap_tok += prev_tok
                else:
                    break
            current_paras = overlap_paras
            current_tokens = overlap_tok

        current_paras.append(p)
        current_tokens += p_tok

    if current_paras:
        windows.append("\n".join(current_paras))

    return windows or [text]


def chunk_document(
    doc: ChunkSource,
    cfg: ChunkingConfig,
    count_tokens_fn: Callable[[str], int] = default_count_tokens,
    split_clauses: bool = True,
) -> tuple[list[Parent], list[Chunk]]:
    """
    Hierarchical legal chunker converting LegalDocument / ChunkSource
    into structured Parents and Chunks.
    """
    clean_text = normalize_whitespace(doc.full_text)
    if not clean_text:
        return [], []

    # Step 2: Inline mode check
    text_len = len(clean_text)
    newline_count = clean_text.count("\n")
    density = (newline_count / text_len) * 1000 if text_len > 0 else 0

    if density < cfg.min_newline_density:
        # Insert newline before structural markers
        clean_text = re.sub(
            r"([;\.\!\?]\s+)(Điều\s+\d+[a-zA-Z]?[\.:]|Chương\s+[IVXLC\d]+|Mục\s+\d+)",
            r"\1\n\2",
            clean_text,
        )

    lines = clean_text.split("\n")

    # Step 3: Scan for structure
    # Detect if document has at least 1 article
    has_articles = any(RE_ARTICLE.match(line.strip()) for line in lines)

    parents: list[Parent] = []
    chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()

    def add_chunk(chunk: Chunk) -> None:
        base_cid = chunk.chunk_id
        cid = base_cid
        dup = 1
        while cid in seen_chunk_ids:
            dup += 1
            cid = f"{base_cid}_{dup}"
        chunk.chunk_id = cid
        seen_chunk_ids.add(cid)
        chunks.append(chunk)

    if not has_articles:
        # Flat chunking mode
        windows = split_text_into_windows(
            clean_text, cfg.flat_window_tokens, cfg.flat_overlap_tokens, count_tokens_fn
        )
        for i, win in enumerate(windows):
            p_id = make_parent_id(doc.doc_id, "F", i)
            tok_count = count_tokens_fn(win)
            parent = Parent(
                parent_id=p_id,
                doc_id=doc.doc_id,
                article_no=None,
                article_title=None,
                breadcrumb=doc.title,
                text=win,
                token_count=tok_count,
                source=doc.source,
                session_id=doc.session_id,
            )
            parents.append(parent)

            chunk = Chunk(
                chunk_id=make_chunk_id(p_id, "flat"),
                parent_id=p_id,
                doc_id=doc.doc_id,
                level="flat",
                article_no=None,
                clause_no=None,
                breadcrumb=doc.title,
                text=win,
                embed_text=f"{doc.title}\n{win}",
                token_count=tok_count,
                content_hash=hash_text(f"{doc.title}\n{win}"),
                source=doc.source,
                session_id=doc.session_id,
            )
            add_chunk(chunk)

        return parents, chunks

    # Structural mode
    current_part = ""
    current_chapter = ""
    current_section = ""

    preamble_lines: list[str] = []
    articles_data: list[dict[str, any]] = []

    current_article: dict[str, any] | None = None
    in_preamble = True

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # Check part
        if RE_PART.match(line):
            current_part = line
            continue

        # Check chapter
        if RE_CHAPTER.match(line):
            # Check if next line contains title
            next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
            if next_line and not (
                RE_PART.match(next_line)
                or RE_CHAPTER.match(next_line)
                or RE_SECTION.match(next_line)
                or RE_ARTICLE.match(next_line)
            ):
                current_chapter = f"{line}: {next_line}"
            else:
                current_chapter = line
            continue

        # Check section
        if RE_SECTION.match(line):
            current_section = line
            continue

        # Check article
        art_match = RE_ARTICLE.match(line)
        if art_match:
            in_preamble = False
            art_no = art_match.group(1)
            art_title = art_match.group(2).strip()

            if current_article is not None:
                articles_data.append(current_article)

            current_article = {
                "article_no": art_no,
                "article_title": art_title,
                "part": current_part,
                "chapter": current_chapter,
                "section": current_section,
                "lines": [line],
            }
            continue

        if in_preamble:
            preamble_lines.append(line)
        elif current_article is not None:
            current_article["lines"].append(line)

    if current_article is not None:
        articles_data.append(current_article)

    # Step 7 (Section 8.2): Non-structured flat fallback
    if not articles_data:
        wins = split_text_into_windows(
            clean_text, cfg.flat_window_tokens, cfg.flat_overlap_tokens, count_tokens_fn
        )
        flat_parents: list[Parent] = []
        flat_chunks: list[Chunk] = []
        for w_idx, win in enumerate(wins):
            w_tok = count_tokens_fn(win)
            p_id = make_parent_id(doc.doc_id, "F", str(w_idx))
            embed_t = f"{doc.title}\n{win}"
            flat_parents.append(
                Parent(
                    parent_id=p_id,
                    doc_id=doc.doc_id,
                    article_no=None,
                    article_title=f"Đoạn {w_idx + 1}",
                    breadcrumb=doc.title,
                    text=win,
                    token_count=w_tok,
                    source=doc.source,
                    session_id=doc.session_id,
                )
            )
            flat_chunks.append(
                Chunk(
                    chunk_id=p_id,
                    parent_id=p_id,
                    doc_id=doc.doc_id,
                    level="flat",
                    article_no=None,
                    clause_no=None,
                    breadcrumb=doc.title,
                    text=win,
                    embed_text=embed_t,
                    token_count=w_tok,
                    content_hash=hash_text(embed_t),
                    source=doc.source,
                    session_id=doc.session_id,
                )
            )
        return flat_parents, flat_chunks

    # 4. Handle Preamble
    if cfg.index_preamble and preamble_lines:
        preamble_text = "\n".join(preamble_lines)
        p_tok = count_tokens_fn(preamble_text)
        if p_tok > cfg.preamble_max_tokens:
            preamble_text = preamble_text[: cfg.preamble_max_tokens * 4]
            p_tok = count_tokens_fn(preamble_text)

        p_id = make_parent_id(doc.doc_id, "P", "0")
        parent = Parent(
            parent_id=p_id,
            doc_id=doc.doc_id,
            article_no=None,
            article_title="Phần mở đầu",
            breadcrumb=doc.title,
            text=preamble_text,
            token_count=p_tok,
            source=doc.source,
            session_id=doc.session_id,
        )
        parents.append(parent)

        add_chunk(
            Chunk(
                chunk_id=make_chunk_id(p_id, "all"),
                parent_id=p_id,
                doc_id=doc.doc_id,
                level="preamble",
                article_no=None,
                clause_no=None,
                breadcrumb=doc.title,
                text=preamble_text,
                embed_text=f"{doc.title}\n{preamble_text}",
                token_count=p_tok,
                content_hash=hash_text(f"{doc.title}\n{preamble_text}"),
                source=doc.source,
                session_id=doc.session_id,
            )
        )

    # 5. Process Articles
    seen_parent_ids = {p.parent_id for p in parents}
    for art in articles_data:
        art_no = art["article_no"]
        art_title = art["article_title"]
        art_text = "\n".join(art["lines"])
        art_tokens = count_tokens_fn(art_text)

        # Assemble Breadcrumb
        bc_parts = [doc.title]
        if art["part"]:
            bc_parts.append(art["part"])
        if art["chapter"]:
            bc_parts.append(art["chapter"])
        if art["section"]:
            bc_parts.append(art["section"])

        title_suffix = f". {art_title}" if art_title else ""
        bc_parts.append(f"Điều {art_no}{title_suffix}")
        breadcrumb = " > ".join(bc_parts)

        base_pid = make_parent_id(doc.doc_id, "A", art_no)
        parent_id = base_pid
        dup_count = 1
        while parent_id in seen_parent_ids:
            dup_count += 1
            parent_id = f"{base_pid}_{dup_count}"
        seen_parent_ids.add(parent_id)
        parent = Parent(
            parent_id=parent_id,
            doc_id=doc.doc_id,
            article_no=art_no,
            article_title=art_title,
            breadcrumb=breadcrumb,
            text=art_text,
            token_count=art_tokens,
            source=doc.source,
            session_id=doc.session_id,
        )
        parents.append(parent)

        # 6. Generate Chunks
        # Parse clauses inside article
        clauses: list[dict[str, any]] = []
        current_clause: dict[str, any] | None = None
        intro_lines: list[str] = []

        for line in art["lines"][1:]:  # skip first line "Điều X"
            cl_match = RE_CLAUSE.match(line)
            if cl_match:
                if current_clause is not None:
                    clauses.append(current_clause)
                current_clause = {
                    "clause_no": cl_match.group(1),
                    "lines": [line],
                }
            elif current_clause is not None:
                current_clause["lines"].append(line)
            else:
                intro_lines.append(line)

        if current_clause is not None:
            clauses.append(current_clause)

        can_split_clauses = (
            split_clauses
            and len(clauses) >= 2
            and art_tokens > cfg.min_article_tokens_for_split
        )

        if can_split_clauses:
            # First clause includes intro lines
            for i, cl in enumerate(clauses):
                cl_lines = list(cl["lines"])
                if i == 0 and intro_lines:
                    cl_lines = intro_lines + cl_lines

                cl_text = "\n".join(cl_lines)
                cl_tok = count_tokens_fn(cl_text)
                cl_no = cl["clause_no"]
                cl_bc = f"{breadcrumb} > Khoản {cl_no}"

                # Merge tiny chunks (< min_chunk_tokens)
                if cl_tok < cfg.min_chunk_tokens and chunks and chunks[-1].parent_id == parent_id:
                    prev_chunk = chunks[-1]
                    merged_text = f"{prev_chunk.text}\n{cl_text}"
                    prev_chunk.text = merged_text
                    prev_chunk.token_count = count_tokens_fn(merged_text)
                    prev_chunk.embed_text = f"{prev_chunk.breadcrumb}\n{merged_text}"
                    prev_chunk.content_hash = hash_text(prev_chunk.embed_text)
                    continue

                if cl_tok > cfg.max_chunk_tokens:
                    # Split clause into windows
                    wins = split_text_into_windows(
                        cl_text, cfg.flat_window_tokens, cfg.flat_overlap_tokens, count_tokens_fn
                    )
                    for w_idx, win in enumerate(wins):
                        w_tok = count_tokens_fn(win)
                        embed_t = f"{cl_bc}\n{win}"
                        add_chunk(
                            Chunk(
                                chunk_id=make_chunk_id(parent_id, f"K{cl_no}:W{w_idx}"),
                                parent_id=parent_id,
                                doc_id=doc.doc_id,
                                level="clause",
                                article_no=art_no,
                                clause_no=cl_no,
                                breadcrumb=cl_bc,
                                text=win,
                                embed_text=embed_t,
                                token_count=w_tok,
                                content_hash=hash_text(embed_t),
                                source=doc.source,
                                session_id=doc.session_id,
                            )
                        )
                else:
                    embed_t = f"{cl_bc}\n{cl_text}"
                    add_chunk(
                        Chunk(
                            chunk_id=make_chunk_id(parent_id, f"K{cl_no}"),
                            parent_id=parent_id,
                            doc_id=doc.doc_id,
                            level="clause",
                            article_no=art_no,
                            clause_no=cl_no,
                            breadcrumb=cl_bc,
                            text=cl_text,
                            embed_text=embed_t,
                            token_count=cl_tok,
                            content_hash=hash_text(embed_t),
                            source=doc.source,
                            session_id=doc.session_id,
                        )
                    )
        else:
            # Whole article chunk
            if art_tokens > cfg.max_chunk_tokens:
                wins = split_text_into_windows(
                    art_text, cfg.flat_window_tokens, cfg.flat_overlap_tokens, count_tokens_fn
                )
                for w_idx, win in enumerate(wins):
                    w_tok = count_tokens_fn(win)
                    embed_t = f"{breadcrumb}\n{win}"
                    add_chunk(
                        Chunk(
                            chunk_id=make_chunk_id(parent_id, f"all:W{w_idx}"),
                            parent_id=parent_id,
                            doc_id=doc.doc_id,
                            level="article",
                            article_no=art_no,
                            clause_no=None,
                            breadcrumb=breadcrumb,
                            text=win,
                            embed_text=embed_t,
                            token_count=w_tok,
                            content_hash=hash_text(embed_t),
                            source=doc.source,
                            session_id=doc.session_id,
                        )
                    )
            else:
                embed_t = f"{breadcrumb}\n{art_text}"
                add_chunk(
                    Chunk(
                        chunk_id=make_chunk_id(parent_id, "all"),
                        parent_id=parent_id,
                        doc_id=doc.doc_id,
                        level="article",
                        article_no=art_no,
                        clause_no=None,
                        breadcrumb=breadcrumb,
                        text=art_text,
                        embed_text=embed_t,
                        token_count=art_tokens,
                        content_hash=hash_text(embed_t),
                        source=doc.source,
                        session_id=doc.session_id,
                    )
                )

    return parents, chunks
