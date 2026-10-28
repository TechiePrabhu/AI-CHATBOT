
import hashlib
import json
import os
import re
from functools import lru_cache

import fitz
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = 850
CHUNK_OVERLAP = 120
REQUEST_TIMEOUT = 180


# ============================================================
# OLLAMA
# ============================================================

def _generate_with_ollama(
    prompt,
    max_output_tokens=700,
    response_format=None,
):
    """Call the local Ollama model and return its response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": max_output_tokens,
            "num_ctx": 4096,
        },
    }

    if response_format:
        payload["format"] = response_format

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        response_data = response.json()
        print("Ollama done_reason:", response_data.get("done_reason"))
        print("Ollama eval_count:", response_data.get("eval_count"))
        print("Ollama eval_duration:", response_data.get("eval_duration"))

        result = response_data.get("response", "").strip()

        if not result:
            raise RuntimeError("Ollama returned an empty response.")

        return result

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Open Ollama and check that "
            f"'{OLLAMA_MODEL}' is installed."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Ollama took too long to respond. Try again or use a "
            "smaller set of PDF files."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.text[:500]
        except Exception:
            pass

        raise RuntimeError(
            f"Ollama API error: {exc}. {detail}"
        ) from exc


def _parse_json_response(raw):
    """Parse JSON even when the model adds surrounding text or fences."""
    if isinstance(raw, (dict, list)):
        return raw

    raw = str(raw).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try the outermost object.
    start = raw.find("{")
    end = raw.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Try the outermost array.
    start = raw.find("[")
    end = raw.rfind("]")

    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The model did not return valid JSON after parsing attempts."
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(filename, pdf_bytes=None):
    """
    Extract text from PDFs.

    Supports:
      extract_pdf_pages("document.pdf", pdf_bytes)
      extract_pdf_pages(uploaded_files)

    When a list of uploaded files is passed, returns (filename, bytes)
    tuples for build_index(), matching the existing app.py interface.
    """
    if pdf_bytes is None and isinstance(filename, (list, tuple)):
        result = []

        for item in filename:
            name = getattr(item, "name", "document.pdf")

            data = (
                item.getvalue()
                if hasattr(item, "getvalue")
                else bytes(item)
            )

            result.append((name, data))

        return result

    if pdf_bytes is None:
        raise ValueError("PDF bytes were not provided.")

    pages = []

    try:
        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "source": str(filename),
                        "page": page_number,
                        "text": text,
                    })

    except Exception as exc:
        raise ValueError(
            f"Could not read '{filename}'. Make sure it is a valid "
            "PDF and is not password-protected."
        ) from exc

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def _split_pages(pages):
    chunks = []

    for page in pages:
        text = re.sub(r"\s+", " ", str(page.get("text", ""))).strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= 30:
                chunks.append({
                    "source": str(page.get("source", "document.pdf")),
                    "page": int(page.get("page", 1)),
                    "text": chunk_text,
                })

            if end >= len(text):
                break

            start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


# ============================================================
# EMBEDDINGS AND INDEX
# ============================================================

@lru_cache(maxsize=1)
def _load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def build_index(uploaded_documents):
    """
    Build one searchable index from multiple PDFs.

    Accepts:
      - Streamlit UploadedFile objects
      - (filename, PDF bytes) tuples
      - already-extracted page dictionaries
    """
    pages = []
    seen_pdf_versions = set()
    source_names = set()

    for document in uploaded_documents:
        # Already-extracted page dictionary.
        if isinstance(document, dict) and "text" in document:
            source = str(document.get("source", "document.pdf"))

            pages.append({
                "source": source,
                "page": int(document.get("page", 1)),
                "text": str(document["text"]),
            })

            source_names.add(source)
            continue

        # Raw PDF tuple or uploaded file.
        if isinstance(document, (tuple, list)) and len(document) >= 2:
            filename, pdf_bytes = document[0], document[1]
        else:
            filename = getattr(document, "name", "document.pdf")
            pdf_bytes = (
                document.getvalue()
                if hasattr(document, "getvalue")
                else bytes(document)
            )

        filename = str(filename)

        if not isinstance(pdf_bytes, (bytes, bytearray)):
            pdf_bytes = bytes(pdf_bytes)

        pdf_bytes = bytes(pdf_bytes)
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()
        pdf_version = (filename, content_hash)

        # Skip only exact duplicate uploads with the same filename.
        if pdf_version in seen_pdf_versions:
            continue

        seen_pdf_versions.add(pdf_version)
        source_names.add(filename)

        pages.extend(extract_pdf_pages(filename, pdf_bytes))

    if not pages:
        raise ValueError(
            "No readable text was found. These may be scanned PDFs; "
            "OCR is not included in this version."
        )

    chunks = _split_pages(pages)

    if not chunks:
        raise ValueError(
            "No usable text chunks could be created from the PDFs."
        )

    model = _load_embedding_model()

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    rag_index = {
        "chunks": chunks,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "model": model,
    }

    stats = {
        "files": len(seen_pdf_versions) if seen_pdf_versions else len(source_names),
        "pages": len(pages),
        "chunks": len(chunks),
        "sources": sorted(source_names),
    }

    return rag_index, stats


# ============================================================
# QUERY PROCESSING AND RELEVANCE
# ============================================================

def _query_terms(text):
    """Get useful terms, keeping technical terms such as EAR and MAR."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was",
        "were", "be", "been", "being", "to", "of", "in", "on", "for",
        "with", "by", "as", "at", "from", "that", "this", "these",
        "those", "it", "its", "what", "which", "who", "when", "where",
        "why", "how", "does", "do", "did", "can", "could", "would",
        "should", "about", "tell", "me", "please", "explain", "describe",
        "give", "show", "find", "according", "document", "pdf",
    }

    return {
        token
        for token in re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*",
            str(text).lower(),
        )
        if len(token) > 1 and token not in stop_words
    }


def _is_comparison_question(question):
    text = str(question).lower()

    phrases = (
        "compare",
        "comparison",
        "difference between",
        "differences between",
        "both documents",
        "both pdfs",
        "across the documents",
        "across both",
        "each document",
        "all documents",
        "all pdfs",
    )

    return any(phrase in text for phrase in phrases)


def _requested_source(question, sources):
    """
    Detect a clearly requested PDF using its filename or common document
    terms. Return a source filename only when the match is reasonably clear.
    """
    q = str(question).lower()
    normalized_q = re.sub(r"[^a-z0-9]+", " ", q).strip()

    # Exact filename or filename stem mentioned by the user.
    for source in sources:
        filename = os.path.basename(str(source)).lower()
        stem = os.path.splitext(filename)[0]

        variants = {
            filename,
            stem,
            re.sub(r"[^a-z0-9]+", " ", filename).strip(),
            re.sub(r"[^a-z0-9]+", " ", stem).strip(),
        }

        for variant in variants:
            if variant and variant in normalized_q:
                return source

    # Common intent-to-document hints for common resume/exam pairs.
    resume_terms = {
        "resume", "résumé", "cv", "my skills", "my education",
        "my projects", "my experience", "my profile",
    }
    exam_terms = {
        "tcs nqt", "nqt", "aptitude paper", "exam paper",
        "question paper", "exam questions", "test paper",
    }

    source_lower = {source: os.path.basename(source).lower() for source in sources}

    if any(term in q for term in resume_terms):
        matches = [
            source for source, name in source_lower.items()
            if "resume" in name or "cv" in name or "curriculum" in name
        ]
        if len(matches) == 1:
            return matches[0]

    if any(term in q for term in exam_terms):
        matches = [
            source for source, name in source_lower.items()
            if "nqt" in name or "exam" in name or "question" in name
            or "paper" in name or "test" in name
        ]
        if len(matches) == 1:
            return matches[0]

    return None


# ============================================================
# MULTI-PDF RETRIEVAL
# ============================================================

def retrieve(question, rag_index, top_k=8):
    """
    Retrieve the best matching passages.

    Important changes:
    - Does NOT force a passage from every PDF into every answer.
    - Respects a clearly requested filename when possible.
    - Includes multiple sources only for comparison/all-document questions.
    - Ranks passages using semantic similarity and keyword overlap.
    """
    question = str(question).strip()
    chunks = rag_index.get("chunks", [])
    embeddings = rag_index.get("embeddings")

    if not question or not chunks or embeddings is None:
        return []

    model = rag_index.get("model") or _load_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    semantic_scores = cosine_similarity(
        question_embedding,
        embeddings,
    )[0]

    query_terms = _query_terms(question)
    lexical_scores = np.zeros(len(chunks), dtype=np.float32)

    if query_terms:
        for i, chunk in enumerate(chunks):
            chunk_terms = _query_terms(chunk.get("text", ""))

            # Fraction of query terms found in this passage.
            overlap = len(query_terms & chunk_terms)
            lexical_scores[i] = overlap / max(1, len(query_terms))

    combined_scores = (
        0.68 * semantic_scores
        + 0.32 * lexical_scores
    )

    sources = sorted({
        str(chunk.get("source", "document.pdf"))
        for chunk in chunks
    })

    requested_source = _requested_source(question, sources)
    comparison = _is_comparison_question(question)

    # If the user clearly asks about one PDF, filter to that PDF.
    candidate_indices = list(range(len(chunks)))

    if requested_source and not comparison:
        candidate_indices = [
            i for i, chunk in enumerate(chunks)
            if str(chunk.get("source", "document.pdf")) == requested_source
        ]

    # Otherwise do not artificially add a low-relevance chunk from
    # every source. Just take the strongest matches.
    ranked_indices = sorted(
        candidate_indices,
        key=lambda i: float(combined_scores[i]),
        reverse=True,
    )

    limit = max(1, min(int(top_k), len(ranked_indices)))
    selected_indices = ranked_indices[:limit]

    # For a comparison question, include a strong match from each source
    # when the source has relevant passages.
    if comparison and len(sources) > 1:
        diversified = []
        used = set()

        for source in sources:
            source_indices = [
                i for i in ranked_indices
                if str(chunks[i].get("source", "document.pdf")) == source
            ]

            if source_indices:
                best_idx = source_indices[0]
                diversified.append(best_idx)
                used.add(best_idx)

        for idx in ranked_indices:
            if idx not in used and len(diversified) < limit:
                diversified.append(idx)
                used.add(idx)

        selected_indices = diversified[:max(limit, len(diversified))]

    results = []

    for idx in selected_indices:
        item = dict(chunks[idx])
        item["score"] = float(combined_scores[idx])
        item["semantic_score"] = float(semantic_scores[idx])
        item["keyword_score"] = float(lexical_scores[idx])
        results.append(item)

    # Do not pass very weak matches to the model as if they were evidence.
    # Keep at least the best result when any text exists, so the answer
    # function can decide whether it has enough evidence.
    return results


def _format_context(chunks, max_chars=7000):
    """Format passages with trustworthy source/page metadata."""
    pieces = []
    used = 0

    for item in chunks:
        label = (
            f"[Source: {item.get('source', 'document.pdf')} | "
            f"Page: {item.get('page', '?')}]"
        )

        piece = f"{label}\n{item.get('text', '')}"
        remaining = max_chars - used

        if remaining <= 0:
            break

        if len(piece) > remaining:
            piece = piece[:remaining]

        pieces.append(piece)
        used += len(piece)

    return "\n\n".join(pieces)


# ============================================================
# PDF QUESTION ANSWERING
# ============================================================

def answer_question(question, rag_index, top_k=5):
    """
    Answer using retrieved PDF evidence only.

    Returns:
      (answer_text, [{"source": filename, "page": page_number}, ...])
    """
    question = str(question).strip()

    if not question:
        return "Please enter a question.", []

    selected = retrieve(
        question,
        rag_index,
        top_k=top_k,
    )

    if not selected:
        return (
            "I couldn't find relevant text in the uploaded PDFs. "
            "Try asking with a more specific keyword or topic.",
            [],
        )

    # If the best result is very weak, avoid asking the model to guess.
    best = selected[0]
    best_score = float(best.get("score", 0.0))
    best_keyword = float(best.get("keyword_score", 0.0))
    best_semantic = float(best.get("semantic_score", 0.0))

    if best_score < 0.18 and best_keyword == 0 and best_semantic < 0.25:
        return (
            "I couldn't find enough relevant information in the uploaded "
            "PDFs to answer that confidently. Try using terms that appear "
            "in the document.",
            [],
        )

    context = _format_context(selected, max_chars=7000)
    comparison = _is_comparison_question(question)

    if comparison:
        scope_instruction = (
            "The user is asking for a comparison or information from "
            "multiple documents. Use evidence from each relevant source "
            "and clearly label which document supports each point."
        )
    else:
        scope_instruction = (
            "Answer the user's specific question. Do not summarize every "
            "document. Ignore excerpts that are unrelated to the question. "
            "Do not mix facts from different documents."
        )

    prompt = f"""You are a precise assistant that answers questions about uploaded PDFs.

STRICT RULES:
1. Use only facts explicitly supported by the PDF excerpts below.
2. Do not use outside knowledge to fill gaps.
3. Do not invent names, skills, dates, exam questions, project details,
   numbers, explanations, or citations.
4. If the evidence does not answer the question, say:
   "The uploaded PDFs do not provide enough information to answer that."
5. Answer the question directly and keep the response concise.
6. Use source names and page numbers only as shown in the excerpts.
7. Never attribute a fact to a PDF unless that exact excerpt supports it.
8. {scope_instruction}

PDF EXCERPTS:
{context}

USER QUESTION:
{question}

Give the answer now. Do not repeat these instructions.
ANSWER:
"""

    answer = _generate_with_ollama(
        prompt,
        max_output_tokens=1000,
    ).strip()

    if not answer:
        answer = (
            "I couldn't produce a reliable answer from the supplied "
            "PDF excerpts. Please try rephrasing your question."
        )

    # Source list is created from retrieved passages, not generated by
    # the language model. Preserve original source/page metadata.
    sources_out = []
    seen = set()

    for item in selected:
        key = (
            str(item.get("source", "document.pdf")),
            int(item.get("page", 1)),
        )

        if key not in seen:
            seen.add(key)
            sources_out.append({
                "source": key[0],
                "page": key[1],
            })

    return answer, sources_out


# ============================================================
# SHARED MULTI-SOURCE HELPERS
# ============================================================

def _group_chunks_by_source(chunks):
    grouped = {}

    for chunk in chunks:
        source = str(chunk.get("source", "document.pdf"))
        grouped.setdefault(source, []).append(chunk)

    return grouped


def _select_chunks_across_sources(chunks, max_chunks=8):
    """
    Round-robin selection for tasks that genuinely need multiple PDFs,
    such as summaries and flashcards. Chat retrieval does not use this.
    """
    if not chunks:
        return []

    grouped = _group_chunks_by_source(chunks)
    sources = list(grouped)
    selected = []
    position = 0
    max_chunks = max(int(max_chunks), len(sources))

    while len(selected) < max_chunks:
        added = False

        for source in sources:
            source_chunks = grouped[source]

            if position < len(source_chunks):
                selected.append(source_chunks[position])
                added = True

                if len(selected) >= max_chunks:
                    break

        if not added:
            break

        position += 1

    return selected


def _chunks_for_source(chunks, max_chunks=5):
    """
    Choose passages spread across a single PDF so long documents are
    represented without flooding the model context with adjacent chunks.
    """
    if len(chunks) <= max_chunks:
        return list(chunks)

    indices = np.linspace(
        0,
        len(chunks) - 1,
        num=max_chunks,
        dtype=int,
    )

    selected = []
    seen = set()

    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            selected.append(chunks[idx])
            seen.add(idx)

    return selected


# ============================================================
# SMART RECALL FLASHCARDS
# ============================================================

def _is_placeholder_flashcard_answer(answer):
    normalized = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        str(answer).lower(),
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    placeholders = {
        "the answer supported by the pdf",
        "answer supported by the pdf",
        "the answer is supported by the pdf",
        "the answer is in the pdf",
        "refer to the pdf",
        "see the pdf",
        "answers may vary",
        "not specified",
        "not provided",
        "the document says so",
    }

    return (
        normalized in placeholders
        or len(normalized.split()) < 3
    )


def _flashcard_context(chunks, max_chars=6500):
    """Build excerpts whose IDs map back to real PDF pages."""
    pieces = []
    selected = []
    used = 0

    for chunk in chunks:
        piece = (
            f"[EXCERPT_ID: {len(selected)}]\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Text: {chunk['text']}"
        )

        remaining = max_chars - used

        if remaining < 250:
            break

        if len(piece) > remaining:
            piece = piece[:remaining]

        pieces.append(piece)
        selected.append(chunk)
        used += len(piece)

        if len(selected) >= 8:
            break

    return "\n\n".join(pieces), selected


def _make_fallback_flashcards(chunks, count):
    """
    Safe fallback when the local model repeatedly returns invalid JSON.
    Answers are copied from the PDF text rather than invented.
    """
    cards = []
    seen = set()

    for chunk in chunks:
        text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()

        # Split into sentence-like units.
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip(" \t-•")
            words = sentence.split()

            if len(words) < 6 or len(words) > 45:
                continue

            normalized = sentence.lower()

            if normalized in seen:
                continue

            # Avoid creating cards from page numbers or fragments.
            if not re.search(r"[a-zA-Z]{3}", sentence):
                continue

            source = str(chunk.get("source", "document.pdf"))
            page = int(chunk.get("page", 1))

            cards.append({
                "front": (
                    f"According to {os.path.basename(source)}, "
                    f"what is stated on page {page}?"
                ),
                "back": sentence,
                "hint1": "Think about the key statement in this passage.",
                "hint2": f"Review page {page} of {os.path.basename(source)}.",
                "source": source,
                "page": page,
            })

            seen.add(normalized)

            if len(cards) >= count:
                return cards

    return cards


def generate_flashcards(rag_index, count=10):
    """
    Create Smart Recall cards.

    Return format remains compatible with app.py:
      {
        "cards": [...],
        "requested": count,
        "complete": bool,
        "message": str
      }
    """
    count = 5 if int(count) <= 5 else 10
    chunks = rag_index.get("chunks", [])

    if not chunks:
        return {
            "cards": [],
            "requested": count,
            "complete": False,
            "message": "No PDF text is available for flashcards.",
        }

    candidates = _select_chunks_across_sources(
        chunks,
        max_chunks=8,
    )
    context, selected = _flashcard_context(candidates)

    if not selected:
        return {
            "cards": [],
            "requested": count,
            "complete": False,
            "message": "No usable PDF excerpts were found.",
        }

    prompt = f"""Create exactly {count} study flashcards using only the excerpts.

Return one valid JSON object and nothing else, in this format:
{{
  "cards": [
    {{
      "front": "A clear question about a fact in the excerpts",
      "back": "The actual factual answer stated in the excerpts",
      "hint1": "A short clue",
      "hint2": "A stronger clue",
      "source_index": 0
    }}
  ]
}}

Rules:
- Use only facts explicitly present in the excerpts.
- Do not invent facts, answers, source names, or page numbers.
- source_index must be an integer matching an EXCERPT_ID.
- Each answer must answer its question directly.
- Do not use placeholders such as "refer to the PDF".
- Make questions distinct and answers concise.
- Return exactly {count} cards if the text supports that many.
- JSON strings must use double quotes.
- Do not add Markdown fences or explanatory text.

EXCERPTS:
{context}
"""

    cards = []
    errors = []

    # Retry once with a simpler instruction if the small model returns
    # malformed JSON or a response that contains no usable cards.
    for attempt in range(2):
        try:
            retry_prompt = prompt

            if attempt == 1:
                retry_prompt += (
                    "\nIMPORTANT RETRY: Return syntactically valid JSON only. "
                    "Keep each card short. The top-level key must be cards."
                )

            raw = _generate_with_ollama(
                retry_prompt,
                max_output_tokens=min(2200, 200 * count),
                response_format="json",
            )

            parsed = _parse_json_response(raw)

            if isinstance(parsed, dict):
                raw_cards = parsed.get("cards", [])
            elif isinstance(parsed, list):
                raw_cards = parsed
            else:
                raw_cards = []

            validated = []
            seen_fronts = set()

            for item in raw_cards:
                if not isinstance(item, dict):
                    continue

                front = str(item.get("front", "")).strip()
                back = str(item.get("back", "")).strip()
                hint1 = str(item.get("hint1", "")).strip()
                hint2 = str(item.get("hint2", "")).strip()

                if len(front) < 5 or len(back.split()) < 3:
                    continue

                if _is_placeholder_flashcard_answer(back):
                    continue

                normalized_front = re.sub(
                    r"\W+",
                    " ",
                    front.lower(),
                ).strip()

                if not normalized_front or normalized_front in seen_fronts:
                    continue

                try:
                    source_index = int(item.get("source_index", 0))
                except (TypeError, ValueError):
                    continue

                if not 0 <= source_index < len(selected):
                    continue

                source_chunk = selected[source_index]

                validated.append({
                    "front": front,
                    "back": back,
                    "hint1": hint1 or "Recall the main concept.",
                    "hint2": hint2 or "Review the related passage in the PDF.",
                    "source": source_chunk["source"],
                    "page": int(source_chunk["page"]),
                })

                seen_fronts.add(normalized_front)

                if len(validated) >= count:
                    break

            if validated:
                cards = validated
                break

            errors.append("The model returned no usable flashcards.")

        except Exception as exc:
            errors.append(str(exc))

    # Fallback avoids an empty flashcard screen if JSON generation fails.
    if len(cards) < count:
        fallback = _make_fallback_flashcards(
            candidates,
            count=count,
        )

        seen = {
            re.sub(r"\W+", " ", card["front"].lower()).strip()
            for card in cards
        }

        for card in fallback:
            normalized = re.sub(
                r"\W+",
                " ",
                card["front"].lower(),
            ).strip()

            if normalized not in seen:
                cards.append(card)
                seen.add(normalized)

            if len(cards) >= count:
                break

    cards = cards[:count]
    complete = len(cards) == count

    if complete:
        message = f"Successfully generated {count} Smart Recall cards."
    elif cards:
        message = (
            f"Generated {len(cards)} of {count} cards from available PDF "
            "text. The PDFs may not contain enough distinct facts for more."
        )
    else:
        detail = errors[-1][:200] if errors else "No usable facts were found."
        message = f"No flashcards could be generated. Details: {detail}"

    return {
        "cards": cards,
        "requested": count,
        "complete": complete,
        "message": message,
    }


# ============================================================
# SUMMARIES
# ============================================================

def generate_summary(rag_index, summary_style="Quick overview"):
    """
    Summarize each uploaded PDF independently, then join the summaries.
    This prevents facts from one PDF being attributed to another.
    """
    chunks = rag_index.get("chunks", [])

    if not chunks:
        return "No PDF text is available to summarize."

    style_instructions = {
        "Quick overview": (
            "Write 5 to 8 concise bullet points covering the main facts."
        ),
        "Detailed summary": (
            "Write a detailed but clear summary with short headings and "
            "bullet points. Include only details supported by the excerpts."
        ),
        "Key points": (
            "List the most important facts, concepts, names, and takeaways."
        ),
        "Study notes": (
            "Create organized study notes with headings and bullet points."
        ),
    }

    instruction = style_instructions.get(
        summary_style,
        style_instructions["Quick overview"],
    )

    grouped = _group_chunks_by_source(chunks)
    summaries = []

    # Separate model calls for separate PDFs: do not mix their content.
    for source, source_chunks in grouped.items():
        selected = _chunks_for_source(
            source_chunks,
            max_chunks=5,
        )

        context = _format_context(
            selected,
            max_chars=5500,
        )

        prompt = f"""Summarize ONLY the PDF named below.

PDF NAME:
{source}

STRICT RULES:
- Use only the supplied excerpts from this PDF.
- Do not use outside knowledge or invent details.
- Do not include facts from any other uploaded document.
- Do not create extra document names or claim that other PDFs were read.
- If the excerpts do not contain a detail, leave it out.
- Do not make up numbers, dates, sections, or examples.
- {instruction}

EXCERPTS FROM {source}:
{context}

SUMMARY OF {source}:
"""

        try:
            summary = _generate_with_ollama(
                prompt,
                max_output_tokens=650,
            ).strip()

            if not summary:
                summary = (
                    "The model did not return a summary for this document."
                )

        except Exception as exc:
            summary = f"Could not summarize this document: {exc}"

        summaries.append(
            f"## {os.path.basename(source)}\n\n{summary}"
        )

    return "\n\n---\n\n".join(summaries)
