
import hashlib
import html
import io

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)

from rag_pipeline import (
    extract_pdf_pages,
    build_index,
    answer_question,
    generate_summary,
    generate_flashcards,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI PDF RAG Assistant")

st.caption(
    "Upload PDFs, ask questions grounded in your documents, "
    "create summaries, and study with Smart Recall flashcards."
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "rag_index": None,
    "index_stats": None,
    "indexed_hash": None,
    "chat_history": [],
    "summary_text": "",
    "summary_style": "Quick overview",
    "flashcard_data": None,
    "flashcard_order": [],
    "flashcard_known": set(),
    "flashcard_revealed": False,
    "recall_confidence": {},
    "recall_hint_level": {},
    "recall_results": {},
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_upload_hash(files):
    """Create a fingerprint for the selected PDF files."""
    digest = hashlib.sha256()

    for file in files or []:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())

    return digest.hexdigest()


def create_summary_pdf(summary):
    """Create a downloadable PDF from the generated summary."""
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph("PDF Summary", styles["Title"]),
        Spacer(1, 18),
    ]

    for line in summary.splitlines():
        line = line.strip()

        if not line:
            story.append(Spacer(1, 8))
            continue

        safe_line = (
            line.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        if line.startswith("#"):
            story.append(
                Paragraph(
                    safe_line.lstrip("# ").strip(),
                    styles["Heading2"],
                )
            )
        else:
            story.append(
                Paragraph(
                    safe_line,
                    styles["BodyText"],
                )
            )

        story.append(Spacer(1, 5))

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


def reset_flashcards():
    """Clear the current flashcard set and its progress."""
    st.session_state.flashcard_data = None
    st.session_state.flashcard_order = []
    st.session_state.flashcard_known = set()
    st.session_state.flashcard_revealed = False

    st.session_state.recall_confidence = {}
    st.session_state.recall_hint_level = {}
    st.session_state.recall_results = {}


def reset_flashcard_progress():
    """Restart the current flashcard set without generating new cards."""
    cards = (
        st.session_state.flashcard_data or {}
    ).get("cards", [])

    st.session_state.flashcard_order = list(range(len(cards)))
    st.session_state.flashcard_known = set()
    st.session_state.flashcard_revealed = False

    st.session_state.recall_confidence = {}
    st.session_state.recall_hint_level = {}
    st.session_state.recall_results = {}


def get_recall_score(results):
    """
    Remembered = 2 points
    Partly remembered = 1 point
    Needs revision = 0 points
    """
    maximum = len(results) * 2

    score = sum(
        int(result.get("points", 0))
        for result in results.values()
    )

    percentage = (
        round((score / maximum) * 100)
        if maximum
        else 0
    )

    return score, maximum, percentage


def get_performance_label(percentage):
    if percentage >= 80:
        return "Good", "🌟"

    if percentage >= 50:
        return "Average", "👍"

    return "Needs Improvement", "📘"


def display_source_pages(sources):
    """Display only source/page information returned by the pipeline."""
    if not sources:
        return

    with st.expander("📌 Source pages"):
        seen = set()

        for source in sources:
            filename = source.get("source", "PDF")
            page = source.get("page", "?")
            key = (filename, page)

            if key in seen:
                continue

            seen.add(key)

            st.write(f"**{filename}** — page {page}")


# ============================================================
# SIDEBAR: UPLOAD AND INDEX
# ============================================================

with st.sidebar:
    st.header("📄 Your documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Text-based PDFs work best. "
            "Scanned image-only PDFs need OCR."
        ),
    )

    current_hash = (
        make_upload_hash(uploaded_files)
        if uploaded_files
        else None
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} PDF file(s) selected."
        )

        if st.button(
            "⚡ Process PDFs",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Reading PDFs and building the search index. "
                "The first run may download the embedding model..."
            ):
                try:
                    documents = extract_pdf_pages(uploaded_files)
                    index, stats = build_index(documents)

                    st.session_state.rag_index = index
                    st.session_state.index_stats = stats
                    st.session_state.indexed_hash = current_hash

                    # Clear outputs associated with the previous index.
                    st.session_state.chat_history = []
                    st.session_state.summary_text = ""

                    reset_flashcards()

                    st.success("PDFs processed successfully.")

                except Exception as exc:
                    st.error(
                        f"Could not process the PDFs: {exc}"
                    )

        if (
            st.session_state.rag_index is not None
            and st.session_state.indexed_hash != current_hash
        ):
            st.warning(
                "The selected PDFs differ from the indexed PDFs. "
                "Click **Process PDFs** to update the index."
            )

    else:
        st.info("Upload PDFs to get started.")

    if st.session_state.index_stats:
        stats = st.session_state.index_stats

        st.divider()
        st.subheader("Index information")

        st.write(f"**Files:** {stats['files']}")
        st.write(f"**Pages with text:** {stats['pages']}")
        st.write(f"**Text chunks:** {stats['chunks']}")

        if stats.get("sources"):
            with st.expander("View indexed filenames"):
                for source in stats["sources"]:
                    st.write(f"• {source}")

    st.divider()
    st.caption("Local generation powered by Ollama.")


# ============================================================
# REQUIRE A PROCESSED INDEX
# ============================================================

if st.session_state.rag_index is None:
    st.info(
        "Upload your PDF files in the sidebar and click "
        "**Process PDFs** to begin."
    )
    st.stop()


# Stop if the selected uploads changed after indexing.
if (
    uploaded_files
    and st.session_state.indexed_hash
    != make_upload_hash(uploaded_files)
):
    st.warning(
        "Your current uploads have changed. Process them before "
        "using the chatbot so answers match the selected PDFs."
    )
    st.stop()


# ============================================================
# MAIN TABS
# ============================================================

chat_tab, summary_tab, flashcard_tab = st.tabs(
    [
        "💬 Chat with PDFs",
        "📝 Summaries",
        "🧠 Smart Recall Flashcards",
    ]
)


# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:
    st.subheader("Ask questions about your PDFs")

    st.caption(
        "Answers use relevant PDF passages. Check the cited "
        "source and page to verify important details."
    )

    # Display previous messages.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                display_source_pages(message["sources"])

    question = st.chat_input(
        "Ask something about your uploaded PDFs..."
    )

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching relevant PDF passages and generating an answer..."
            ):
                try:
                    answer, sources = answer_question(
                        question,
                        st.session_state.rag_index,
                        top_k=5,
                    )

                except Exception as exc:
                    answer = (
                        f"Could not generate an answer: {exc}"
                    )
                    sources = []

            st.markdown(answer)

            if sources:
                display_source_pages(sources)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ============================================================
# SUMMARY TAB
# ============================================================

with summary_tab:
    st.subheader("Generate a PDF summary")

    summary_options = [
        "Quick overview",
        "Detailed summary",
        "Key points",
        "Study notes",
    ]

    previous_style = st.session_state.summary_style

    try:
        default_style_index = summary_options.index(previous_style)
    except ValueError:
        default_style_index = 0

    summary_style = st.selectbox(
        "Choose summary style",
        summary_options,
        index=default_style_index,
        key="summary_style_selector",
    )

    if st.button(
        "✨ Generate summary",
        type="primary",
    ):
        with st.spinner(
            "Generating separate summaries for your PDFs..."
        ):
            try:
                result = generate_summary(
                    st.session_state.rag_index,
                    summary_style=summary_style,
                )

                st.session_state.summary_text = result
                st.session_state.summary_style = summary_style

            except Exception as exc:
                st.session_state.summary_text = (
                    f"Could not generate summary: {exc}"
                )

    if st.session_state.summary_text:
        st.markdown(st.session_state.summary_text)

        try:
            pdf_bytes = create_summary_pdf(
                st.session_state.summary_text
            )

            st.download_button(
                "⬇️ Download summary as PDF",
                data=pdf_bytes,
                file_name="pdf_summary.pdf",
                mime="application/pdf",
            )

        except Exception as exc:
            st.warning(
                f"Could not create the downloadable PDF: {exc}"
            )


# ============================================================
# SMART RECALL FLASHCARDS
# ============================================================

with flashcard_tab:
    st.subheader("🧠 Smart Recall Mode")

    st.write(
        "Try to recall the answer before revealing it. "
        "Rate your confidence, use hints when needed, and track "
        "your study score."
    )

    st.caption(
        "Good = confident, Average = somewhat sure, "
        "Bad = unsure. Your final score is based on your self-rating "
        "after checking each answer."
    )

    flashcard_data = st.session_state.flashcard_data

    # --------------------------------------------------------
    # GENERATE A NEW SET
    # --------------------------------------------------------

    if flashcard_data is None:
        st.info(
            "Create a set of study cards from your uploaded PDFs."
        )

        flashcard_count = st.selectbox(
            "Number of flashcards",
            options=[5, 10],
            index=1,
            key="flashcard_count_select",
        )

        if st.button(
            "✨ Generate Smart Recall cards",
            type="primary",
        ):
            with st.spinner(
                "Creating flashcards from your PDFs using Ollama..."
            ):
                try:
                    data = generate_flashcards(
                        st.session_state.rag_index,
                        count=flashcard_count,
                    )

                    st.session_state.flashcard_data = data

                    st.session_state.flashcard_order = list(
                        range(len(data.get("cards", [])))
                    )

                    st.session_state.flashcard_known = set()
                    st.session_state.flashcard_revealed = False
                    st.session_state.recall_confidence = {}
                    st.session_state.recall_hint_level = {}
                    st.session_state.recall_results = {}

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Could not generate flashcards: {exc}"
                    )

    # --------------------------------------------------------
    # DISPLAY THE CURRENT SET
    # --------------------------------------------------------

    else:
        cards = flashcard_data.get("cards", [])
        card_order = st.session_state.flashcard_order
        results = st.session_state.recall_results

        if not cards:
            st.warning(
                flashcard_data.get(
                    "message",
                    "No flashcards were generated.",
                )
            )

            if st.button("🔄 Try generating again"):
                reset_flashcards()
                st.rerun()

        else:
            total_cards = len(cards)
            completed_count = len(results)

            st.caption(
                flashcard_data.get(
                    "message",
                    f"Loaded {total_cards} flashcards.",
                )
            )

            # ------------------------------------------------
            # PROGRESS METRICS
            # ------------------------------------------------

            score, maximum_score, percentage = get_recall_score(
                results
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Total cards",
                total_cards,
            )

            metric2.metric(
                "Completed",
                completed_count,
            )

            metric3.metric(
                "Recall score",
                f"{score}/{maximum_score}",
            )

            st.progress(
                completed_count / total_cards
                if total_cards
                else 0.0
            )

            if completed_count:
                st.caption(
                    f"Current score: {percentage}% "
                    f"({completed_count} card(s) completed)"
                )

            st.divider()

            # ------------------------------------------------
            # FINAL RESULTS
            # ------------------------------------------------

            if not card_order:
                label, emoji = get_performance_label(percentage)

                st.success(
                    "🎉 You completed this Smart Recall session!"
                )

                result_col1, result_col2, result_col3 = st.columns(3)

                result_col1.metric(
                    "Final score",
                    f"{score}/{maximum_score}",
                )

                result_col2.metric(
                    "Percentage",
                    f"{percentage}%",
                )

                result_col3.metric(
                    "Performance",
                    f"{emoji} {label}",
                )

                remembered_count = sum(
                    1
                    for item in results.values()
                    if item.get("rating") == "Remembered"
                )

                partial_count = sum(
                    1
                    for item in results.values()
                    if item.get("rating") == "Partly remembered"
                )

                revise_count = sum(
                    1
                    for item in results.values()
                    if item.get("rating") == "Needs revision"
                )

                st.write("### Your study report")

                report_col1, report_col2, report_col3 = st.columns(3)

                report_col1.metric(
                    "Remembered",
                    remembered_count,
                )

                report_col2.metric(
                    "Partly remembered",
                    partial_count,
                )

                report_col3.metric(
                    "Needs revision",
                    revise_count,
                )

                st.info(
                    "Good: 80–100%. Average: 50–79%. "
                    "Needs Improvement: below 50%. "
                    "This is a self-assessment score, not an "
                    "independently graded knowledge test."
                )

                st.write("### Review your results")

                for result in results.values():
                    with st.expander(
                        f"{result['rating']}: {result['front']}"
                    ):
                        st.write(
                            f"**Answer:** {result['back']}"
                        )

                        st.caption(
                            f"Source: {result['source']} — "
                            f"page {result['page']}"
                        )

                        st.caption(
                            "Confidence before reveal: "
                            f"{result.get('confidence', 'Not selected')}"
                        )

                st.divider()

                restart_col, new_col = st.columns(2)

                with restart_col:
                    if st.button(
                        "🔁 Restart this set",
                        use_container_width=True,
                    ):
                        reset_flashcard_progress()
                        st.rerun()

                with new_col:
                    if st.button(
                        "✨ Generate a new set",
                        type="primary",
                        use_container_width=True,
                    ):
                        reset_flashcards()
                        st.rerun()

            # ------------------------------------------------
            # CURRENT FLASHCARD
            # ------------------------------------------------

            else:
                current_index = card_order[0]
                card = cards[current_index]

                confidence_map = st.session_state.recall_confidence
                hint_map = st.session_state.recall_hint_level

                confidence = confidence_map.get(current_index)
                hint_level = hint_map.get(current_index, 0)

                st.caption(
                    f"Card {completed_count + 1} of {total_cards}"
                )

                # --------------------------------------------
                # CONFIDENCE RATING
                # --------------------------------------------

                st.write("### 1. Rate your confidence")

                confidence_cols = st.columns(3)

                with confidence_cols[0]:
                    if st.button(
                        "🟢 Good",
                        type=(
                            "primary"
                            if confidence == "Good"
                            else "secondary"
                        ),
                        use_container_width=True,
                        key=f"confidence_good_{current_index}",
                    ):
                        confidence_map[current_index] = "Good"
                        st.rerun()

                with confidence_cols[1]:
                    if st.button(
                        "🟡 Average",
                        type=(
                            "primary"
                            if confidence == "Average"
                            else "secondary"
                        ),
                        use_container_width=True,
                        key=f"confidence_average_{current_index}",
                    ):
                        confidence_map[current_index] = "Average"
                        st.rerun()

                with confidence_cols[2]:
                    if st.button(
                        "🔴 Bad",
                        type=(
                            "primary"
                            if confidence == "Bad"
                            else "secondary"
                        ),
                        use_container_width=True,
                        key=f"confidence_bad_{current_index}",
                    ):
                        confidence_map[current_index] = "Bad"
                        st.rerun()

                if confidence:
                    st.caption(
                        f"Your confidence rating: **{confidence}**"
                    )
                else:
                    st.caption(
                        "Choose a confidence rating before revealing "
                        "the answer."
                    )

                # --------------------------------------------
                # QUESTION AND ANSWER
                # --------------------------------------------

                st.write("### 2. Recall the answer")

                safe_front = html.escape(
                    str(card.get("front", ""))
                )

                safe_back = html.escape(
                    str(card.get("back", ""))
                )

                with st.container(border=True):
                    st.markdown(
                        "<div style='text-align:center; "
                        "font-size:12px; letter-spacing:2px; "
                        "color:#777;'>QUESTION</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f"<div style='text-align:center; "
                        f"font-size:24px; font-weight:600; "
                        f"padding:22px 8px;'>{safe_front}</div>",
                        unsafe_allow_html=True,
                    )

                    if st.session_state.flashcard_revealed:
                        st.divider()

                        st.markdown(
                            "<div style='text-align:center; "
                            "font-size:12px; letter-spacing:2px; "
                            "color:#777;'>ANSWER</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"<div style='text-align:center; "
                            f"font-size:18px; padding:14px 8px;'>"
                            f"{safe_back}</div>",
                            unsafe_allow_html=True,
                        )

                    else:
                        st.caption(
                            "Try to answer from memory before looking "
                            "at hints."
                        )

                st.caption(
                    f"📌 Source: {card.get('source', 'PDF')} — "
                    f"page {card.get('page', '?')}"
                )

                # --------------------------------------------
                # PROGRESSIVE HINTS
                # --------------------------------------------

                if not st.session_state.flashcard_revealed:
                    st.write("### 3. Need help? Reveal a hint")

                    if hint_level >= 1:
                        hint1 = html.escape(
                            str(
                                card.get(
                                    "hint1",
                                    "Think about the main concept.",
                                )
                            )
                        )

                        st.info(f"💡 Hint 1: {hint1}")

                    if hint_level >= 2:
                        hint2 = html.escape(
                            str(
                                card.get(
                                    "hint2",
                                    "Review the related section of your PDF.",
                                )
                            )
                        )

                        st.info(f"🔎 Hint 2: {hint2}")

                    hint_col, reveal_col = st.columns(2)

                    with hint_col:
                        if hint_level < 2:
                            button_label = (
                                "💡 Show hint 1"
                                if hint_level == 0
                                else "🔎 Show hint 2"
                            )

                            if st.button(
                                button_label,
                                use_container_width=True,
                                key=f"hint_{current_index}_{hint_level}",
                            ):
                                hint_map[current_index] = hint_level + 1
                                st.rerun()

                    with reveal_col:
                        if st.button(
                            "👀 Reveal answer",
                            type="primary",
                            use_container_width=True,
                            key=f"reveal_{current_index}",
                        ):
                            if not confidence:
                                st.warning(
                                    "Choose Good, Average, or Bad "
                                    "confidence first."
                                )
                            else:
                                st.session_state.flashcard_revealed = True
                                st.rerun()

                # --------------------------------------------
                # SELF-ASSESSMENT
                # --------------------------------------------

                else:
                    st.write("### 4. How well did you recall it?")

                    st.caption(
                        "Be honest with yourself. Use your answer "
                        "before revealing as the basis for your rating."
                    )

                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        if st.button(
                            "✅ Remembered",
                            type="primary",
                            use_container_width=True,
                            key=f"remembered_{current_index}",
                        ):
                            results[current_index] = {
                                "front": card.get("front", ""),
                                "back": card.get("back", ""),
                                "source": card.get("source", "PDF"),
                                "page": card.get("page", "?"),
                                "confidence": confidence or "Not selected",
                                "rating": "Remembered",
                                "points": 2,
                            }

                            st.session_state.flashcard_known.add(
                                current_index
                            )

                            st.session_state.flashcard_order = card_order[1:]
                            st.session_state.flashcard_revealed = False
                            st.rerun()

                    with result_col2:
                        if st.button(
                            "🟡 Partly remembered",
                            use_container_width=True,
                            key=f"partly_{current_index}",
                        ):
                            results[current_index] = {
                                "front": card.get("front", ""),
                                "back": card.get("back", ""),
                                "source": card.get("source", "PDF"),
                                "page": card.get("page", "?"),
                                "confidence": confidence or "Not selected",
                                "rating": "Partly remembered",
                                "points": 1,
                            }

                            st.session_state.flashcard_order = card_order[1:]
                            st.session_state.flashcard_revealed = False
                            st.rerun()

                    with result_col3:
                        if st.button(
                            "🔁 Needs revision",
                            use_container_width=True,
                            key=f"revise_{current_index}",
                        ):
                            results[current_index] = {
                                "front": card.get("front", ""),
                                "back": card.get("back", ""),
                                "source": card.get("source", "PDF"),
                                "page": card.get("page", "?"),
                                "confidence": confidence or "Not selected",
                                "rating": "Needs revision",
                                "points": 0,
                            }

                            st.session_state.flashcard_order = card_order[1:]
                            st.session_state.flashcard_revealed = False
                            st.rerun()

                # --------------------------------------------
                # CLEAR CURRENT SET
                # --------------------------------------------

                st.divider()

                if st.button(
                    "🗑️ Clear cards and create a new set",
                    key="clear_flashcards",
                ):
                    reset_flashcards()
                    st.rerun()
