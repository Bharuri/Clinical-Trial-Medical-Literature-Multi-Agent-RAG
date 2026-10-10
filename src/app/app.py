"""
Streamlit UI for the Clinical Trial & Medical Literature Multi-Agent RAG.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -m streamlit run src/app/app.py

Credentials and queries live only in st.session_state. Nothing is written to
disk, cookies, or local storage, except answers the user explicitly accepts,
which are saved to the accepted-answers DuckDB file.
"""

from typing import List, Optional

import duckdb
import streamlit as st

from src.app.accepted_answers import load_accepted_answers, save_accepted_answer
from src.app.chunk_browser import list_chunk_sources, load_chunks
from src.app.service import answer_question
from src.ingestion.embeddings import EmbeddingError
from src.retrieval_agent.bedrock_client import BedrockError
from src.retrieval_agent.fact_check import FactCheckResult
from src.retrieval_agent.retrieval import RetrievalError, RetrievalResult
from src.retrieval_agent.run import DraftAttempt, PipelineResult
from src.utils.config import Config, ConfigurationError

PIPELINE_ERRORS = (BedrockError, RetrievalError, EmbeddingError, ConfigurationError, ValueError)

KEY_ACCESS = "aws_access_key_id"
KEY_SECRET = "aws_secret_access_key"
KEY_REGION = "aws_region_name"
KEY_MODEL = "bedrock_model_id"
KEY_QUESTION = "research_question"
KEY_RESULT = "pipeline_result"
KEY_ERROR = "pipeline_error"
KEY_ACCEPTED_PREFIX = "accepted_answer_id_"
KEY_ACCEPT_ERROR_PREFIX = "accept_error_"


def reset_session() -> None:
    """Clear all widget values and results. Runs as the button's on_click callback."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def region_options(config: Config) -> List[str]:
    """Configured regions, with the default region included even if missing from the list."""
    default = config.get_default_region()
    regions = config.get_bedrock_regions()
    return regions if default in regions else [default] + regions


def render_sidebar(config: Config) -> None:
    with st.sidebar:
        st.header("🔐 Secure AWS Configuration")
        st.text_input("AWS Access Key ID", key=KEY_ACCESS)
        st.text_input("AWS Secret Access Key", type="password", key=KEY_SECRET)
        st.selectbox("AWS Region", options=region_options(config), key=KEY_REGION)
        st.text_input(
            "Bedrock Model ID",
            value=config.get_default_bedrock_model(),
            key=KEY_MODEL,
        )
        st.button(
            "Clear Credentials / Reset Session",
            on_click=reset_session,
            use_container_width=True,
        )


def run_pipeline_from_inputs() -> None:
    """Call the service with the current inputs and store the outcome in session state."""
    st.session_state.pop(KEY_RESULT, None)
    st.session_state.pop(KEY_ERROR, None)
    with st.spinner("Routing, retrieving, drafting, and fact-checking..."):
        try:
            st.session_state[KEY_RESULT] = answer_question(
                st.session_state.get(KEY_QUESTION, ""),
                aws_access_key_id=st.session_state.get(KEY_ACCESS, ""),
                aws_secret_access_key=st.session_state.get(KEY_SECRET, ""),
                aws_region_name=st.session_state.get(KEY_REGION),
                model_id=st.session_state.get(KEY_MODEL, ""),
            )
        except PIPELINE_ERRORS as e:
            st.session_state[KEY_ERROR] = str(e)


def render_retrieval(retrieval: Optional[RetrievalResult]) -> None:
    with st.expander("1. Router & Retrieval Agent", expanded=True):
        if retrieval is None:
            st.info("Retrieval did not run.")
            return

        col_route, col_dims, col_matches = st.columns(3)
        col_route.metric("Route", retrieval.route)
        col_dims.metric("Query vector dimensions", retrieval.query_dimensions)
        col_matches.metric("HNSW matches", retrieval.match_count)

        if not retrieval.chunks:
            st.warning("No chunks matched. Check that ingestion has populated the database.")
            return

        st.dataframe(
            [
                {
                    "Rank": rank,
                    "Similarity": round(chunk.similarity_score, 4),
                    "Source file": chunk.source_file,
                    "Page": chunk.page_number,
                    "Type": chunk.chunk_type,
                    "Section": chunk.section_header,
                    "Preview": chunk.preview(200),
                }
                for rank, chunk in enumerate(retrieval.chunks, start=1)
            ]
        )


def version_label(attempt: DraftAttempt) -> str:
    status = "✅ Approved" if attempt.fact_check.is_approved else "❌ Not approved"
    return f"{attempt.label}: {status}"


def render_synthesis(result: PipelineResult) -> None:
    with st.expander("2. Synthesis & Drafting Agent", expanded=True):
        if not result.attempts:
            st.info("No draft was produced because no chunks were retrieved.")
            return
        st.write(
            f"Drafting finished. Revision rounds: {result.revisions}. "
            "Each version is in its own tab."
        )
        tabs = st.tabs([attempt.label for attempt in result.attempts])
        for tab, attempt in zip(tabs, result.attempts):
            with tab:
                st.markdown(attempt.draft)


def render_check_details(check: FactCheckResult) -> None:
    if check.is_approved:
        st.success("Approved by the fact-checker.")
    else:
        st.error("Not approved. This version was sent back for revision, or the revision limit was reached.")

    if check.critique:
        st.write("**Critique**")
        st.write(check.critique)
    if check.issues:
        st.write("**Issues**")
        for issue in check.issues:
            st.markdown(f"- {issue}")

    st.write("**Verification logs**")
    st.code("\n".join(check.logs), language=None)


def accept_attempt(question: str, answer: str, index: int) -> None:
    """Save an attempt the user accepted. Runs as the button's on_click callback."""
    st.session_state.pop(f"{KEY_ACCEPT_ERROR_PREFIX}{index}", None)
    try:
        st.session_state[f"{KEY_ACCEPTED_PREFIX}{index}"] = save_accepted_answer(question, answer)
    except (ValueError, ConfigurationError, OSError, duckdb.Error) as e:
        st.session_state[f"{KEY_ACCEPT_ERROR_PREFIX}{index}"] = str(e)


def render_accept_control(question: str, answer: str, index: int) -> None:
    accepted_id = st.session_state.get(f"{KEY_ACCEPTED_PREFIX}{index}")
    if accepted_id is not None:
        st.success(f"Accepted answer saved (id {accepted_id}).")
        return

    st.button(
        "Accept this answer",
        key=f"accept_button_{index}",
        on_click=accept_attempt,
        args=(question, answer, index),
        help="Save this question and this version of the answer to the accepted-answers database.",
    )
    error = st.session_state.get(f"{KEY_ACCEPT_ERROR_PREFIX}{index}")
    if error:
        st.error(f"Could not save the accepted answer: {error}")


def render_fact_check(result: PipelineResult) -> None:
    with st.expander("3. Critic & Fact-Checking Agent", expanded=True):
        if not result.attempts:
            st.info("Fact-checking did not run.")
            return

        tabs = st.tabs([version_label(attempt) for attempt in result.attempts])
        for index, (tab, attempt) in enumerate(zip(tabs, result.attempts)):
            with tab:
                render_check_details(attempt.fact_check)
                render_accept_control(result.question, attempt.draft, index)

        st.write("**Pipeline logs**")
        st.code("\n".join(result.logs), language=None)


def render_final(result: PipelineResult) -> None:
    st.subheader("✅ Final Verified Answer")
    if result.final_answer:
        st.markdown(result.final_answer)
    elif result.draft is None:
        st.info("No verified answer. No matching chunks were found for this question.")
    else:
        st.info(
            f"No verified answer. The fact-checker did not approve the draft after "
            f"{result.revisions} revision round(s). Review the critique above."
        )


def render_results() -> None:
    error = st.session_state.get(KEY_ERROR)
    if error:
        st.error(error)
        return

    result: Optional[PipelineResult] = st.session_state.get(KEY_RESULT)
    if result is None:
        return

    render_retrieval(result.retrieval)
    render_synthesis(result)
    render_fact_check(result)
    render_final(result)


def render_accepted_answers() -> None:
    st.subheader("Accepted Answers")
    keyword = st.text_input(
        "Filter by text in the question or answer:",
        key="accepted_filter",
    )
    rows = load_accepted_answers(keyword)
    if not rows:
        st.info("No accepted answers match." if keyword.strip() else "No answers have been accepted yet.")
        return

    st.caption(f"{len(rows)} accepted answer(s), newest first.")
    for row in rows:
        title = row["question"] if len(row["question"]) <= 120 else row["question"][:117] + "..."
        with st.expander(f"#{row['id']} · {title}"):
            st.write(f"**Question:** {row['question']}")
            st.caption(f"Accepted at {row['created_at']}")
            st.markdown(row["answer"])


ALL_SOURCES_LABEL = "All sources"


def render_chunk_table() -> None:
    st.subheader("Chunk Table")
    try:
        sources = list_chunk_sources()
        keyword_col, source_col = st.columns([3, 2])
        with keyword_col:
            keyword = st.text_input("Filter by text in the chunk content:", key="chunk_filter")
        with source_col:
            source = st.selectbox("Source file:", [ALL_SOURCES_LABEL, *sources], key="chunk_source")
        source_file = "" if source == ALL_SOURCES_LABEL else source
        rows, total = load_chunks(keyword, source_file)
    except duckdb.Error as e:
        st.warning(
            f"Could not read the chunk table: {e}. "
            "If another process (for example, an older app session or the ingestion job) "
            "has the RAG database open, stop it and refresh this page."
        )
        return

    if not total:
        filtered = bool(keyword.strip() or source_file)
        st.info(
            "No chunks match the filters."
            if filtered
            else "The chunk table is empty or missing. Run the ingestion job first."
        )
        return

    shown = f" Showing the first {len(rows)}." if total > len(rows) else ""
    st.caption(f"{total} matching chunk(s).{shown}")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Clinical Trial & Medical RAG", page_icon="🧬", layout="wide")
    config = Config.get_config()

    st.title("🧬 Clinical Trial & Medical Literature Multi-Agent RAG")
    st.caption(
        "Answers from your PDF library using DuckDB vector search, "
        "1,024-dimensional Sentence Transformers embeddings, and Amazon Bedrock."
    )

    render_sidebar(config)

    pipeline_tab, accepted_tab, chunks_tab = st.tabs(
        ["Ask & Verify", "Accepted Answers", "Chunk Table"]
    )
    with pipeline_tab:
        st.text_input(
            "Enter your clinical trial / medical research question:",
            key=KEY_QUESTION,
        )
        if st.button("Run Multi-Agent Pipeline", type="primary"):
            run_pipeline_from_inputs()

        render_results()
    with accepted_tab:
        render_accepted_answers()
    with chunks_tab:
        render_chunk_table()


main()
