"""
Streamlit UI tests using AppTest. The pipeline service is patched, so no
embedding model is loaded and no Bedrock call is made.
"""

from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
from streamlit.testing.v1 import AppTest

from src.retrieval_agent.bedrock_client import BedrockError
from src.retrieval_agent.fact_check import FactCheckResult
from src.retrieval_agent.retrieval import RetrievalResult, RetrievedChunk
from src.retrieval_agent.run import DraftAttempt, PipelineResult
from src.utils.config import Config

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app" / "app.py")
SERVICE_TARGET = "src.app.service.answer_question"

CHUNK = RetrievedChunk(
    chunk_id="c1",
    source_file="trial.pdf",
    page_number=2,
    chunk_type="text",
    section_header="Results",
    content="The hazard ratio was 0.75 across the cohort.",
    similarity_score=0.92,
)


def approved_result() -> PipelineResult:
    check = FactCheckResult(is_approved=True, critique="", issues=[], logs=["approved"])
    draft = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
    return PipelineResult(
        question="What was the hazard ratio?",
        retrieval=RetrievalResult(
            question="What was the hazard ratio?",
            route="text",
            query_dimensions=768,
            chunks=[CHUNK],
        ),
        draft=draft,
        fact_check=check,
        revisions=0,
        final_answer=draft,
        logs=["approved"],
        attempts=[DraftAttempt("Initial draft", draft, check)],
    )


def rejected_result() -> PipelineResult:
    first = FactCheckResult(
        is_approved=False,
        critique="Number does not match the source.",
        issues=["Unsupported number: 0.9 does not appear in any source"],
        logs=["rejected 0"],
    )
    last = FactCheckResult(
        is_approved=False,
        critique="Number does not match the source.",
        issues=["Unsupported number: 0.8 does not appear in any source"],
        logs=["rejected 1"],
    )
    return PipelineResult(
        question="What was the hazard ratio?",
        retrieval=RetrievalResult(
            question="What was the hazard ratio?",
            route="text",
            query_dimensions=768,
            chunks=[CHUNK],
        ),
        draft="The hazard ratio was 0.8.",
        fact_check=last,
        revisions=2,
        final_answer=None,
        logs=["rejected"],
        attempts=[
            DraftAttempt("Initial draft", "The hazard ratio was 0.9.", first),
            DraftAttempt("Revision 1", "The hazard ratio was 0.8.", last),
        ],
    )


def revised_result() -> PipelineResult:
    first = FactCheckResult(
        is_approved=False,
        critique="Remove the uncited sentence.",
        issues=['Uncited claim: "The drug works well."'],
        logs=["rejected 0"],
    )
    approved = FactCheckResult(is_approved=True, critique="", issues=[], logs=["approved 1"])
    final = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
    return PipelineResult(
        question="What was the hazard ratio?",
        retrieval=RetrievalResult(
            question="What was the hazard ratio?",
            route="text",
            query_dimensions=768,
            chunks=[CHUNK],
        ),
        draft=final,
        fact_check=approved,
        revisions=1,
        final_answer=final,
        logs=["revised"],
        attempts=[
            DraftAttempt("Initial draft", "The drug works well. The hazard ratio was 0.75.", first),
            DraftAttempt("Revision 1", final, approved),
        ],
    )


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


@pytest.fixture(autouse=True)
def no_real_chunk_database():
    """Keep UI tests off the real RAG database. Tests that need chunk data patch it again."""
    with patch(CHUNK_SOURCES_TARGET, return_value=[]), patch(CHUNK_LOAD_TARGET, return_value=([], 0)):
        yield


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def run_button(at: AppTest):
    return next(b for b in at.button if b.label == "Run Multi-Agent Pipeline")


def reset_button(at: AppTest):
    return next(b for b in at.sidebar.button if "Reset" in b.label)


def fill_and_run(at: AppTest, question: str = "What was the hazard ratio?") -> AppTest:
    at.text_input(key="aws_access_key_id").input("AKIATEST")
    at.text_input(key="aws_secret_access_key").input("secret")
    at.text_input(key="research_question").input(question)
    run_button(at).click()
    at.run()
    return at


def test_page_renders_title_and_sidebar_widgets(app):
    assert not app.exception
    assert "Multi-Agent RAG" in app.title[0].value
    labels = [w.label for w in app.sidebar.text_input]
    assert "AWS Access Key ID" in labels
    assert "AWS Secret Access Key" in labels
    assert "Bedrock Model ID" in labels


def test_secret_key_is_masked(app):
    secret = next(w for w in app.sidebar.text_input if w.label == "AWS Secret Access Key")
    assert secret.proto.type == TextInputProto.PASSWORD


def test_region_options_come_from_config(app):
    region = next(w for w in app.sidebar.selectbox if w.label == "AWS Region")
    config = Config.get_config()
    assert region.value == config.get_default_region()
    assert set(config.get_bedrock_regions()) <= set(region.options)


def test_empty_question_shows_error_without_loading_model(app):
    with patch("src.app.service.get_query_encoder") as encoder_factory:
        run_button(app).click()
        app.run()

    assert not app.exception
    assert any("research question" in e.value for e in app.error)
    encoder_factory.assert_not_called()


def test_approved_answer_is_shown_as_final(app):
    with patch(SERVICE_TARGET, return_value=approved_result()) as service:
        fill_and_run(app)

    assert not app.exception
    service.assert_called_once()
    assert service.call_args.args[0] == "What was the hazard ratio?"
    assert service.call_args.kwargs["aws_access_key_id"] == "AKIATEST"
    assert any("Approved by the fact-checker" in s.value for s in app.success)
    markdown = " ".join(m.value for m in app.markdown)
    assert "The hazard ratio was 0.75" in markdown
    assert any("Final Verified Answer" in h.value for h in app.subheader)


def test_rejected_draft_is_not_shown_as_final(app):
    with patch(SERVICE_TARGET, return_value=rejected_result()):
        fill_and_run(app)

    assert not app.exception
    assert any("Not approved" in e.value for e in app.error)
    assert any("did not approve the draft after 2 revision" in i.value for i in app.info)
    # The draft appears once, in the drafting expander. It must not be repeated as the final answer.
    draft_mentions = sum(m.value.count("hazard ratio was 0.9") for m in app.markdown)
    assert draft_mentions == 1


def test_bedrock_error_is_displayed(app):
    with patch(SERVICE_TARGET, side_effect=BedrockError("Access key and secret are required.")):
        fill_and_run(app)

    assert not app.exception
    assert any("Access key and secret are required." in e.value for e in app.error)


def test_retrieval_details_and_logs_are_rendered(app):
    with patch(SERVICE_TARGET, return_value=approved_result()):
        fill_and_run(app)

    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Query vector dimensions"] == "768"
    assert metrics["HNSW matches"] == "1"
    assert metrics["Route"] == "text"

    table = app.dataframe[0].value
    row = table.iloc[0]
    assert row["Source file"] == "trial.pdf"
    assert row["Page"] == 2
    assert row["Similarity"] == 0.92
    assert "hazard ratio was 0.75" in row["Preview"]

    assert any("approved" in c.value for c in app.code)


def test_reset_clears_credentials_and_results(app):
    with patch(SERVICE_TARGET, return_value=approved_result()):
        fill_and_run(app)

    reset_button(app).click()
    app.run()

    assert not app.exception
    access = next(w for w in app.sidebar.text_input if w.label == "AWS Access Key ID")
    secret = next(w for w in app.sidebar.text_input if w.label == "AWS Secret Access Key")
    assert access.value == ""
    assert secret.value == ""
    assert not any("Final Verified Answer" in h.value for h in app.subheader)


def test_both_versions_and_their_fact_checks_are_shown(app):
    with patch(SERVICE_TARGET, return_value=revised_result()):
        fill_and_run(app)

    assert not app.exception
    markdown = " ".join(m.value for m in app.markdown)
    assert "The drug works well." in markdown
    assert "hazard ratio was 0.75" in markdown

    assert any("Not approved" in e.value for e in app.error)
    assert any("Approved by the fact-checker" in s.value for s in app.success)
    assert any('Uncited claim: "The drug works well."' in m.value for m in app.markdown)
    assert any("Final Verified Answer" in h.value for h in app.subheader)


def test_rejected_run_shows_each_version_once(app):
    with patch(SERVICE_TARGET, return_value=rejected_result()):
        fill_and_run(app)

    assert not app.exception
    markdown = " ".join(m.value for m in app.markdown)
    assert markdown.count("hazard ratio was 0.9") == 1
    assert markdown.count("hazard ratio was 0.8") == 1
    assert any("Unsupported number: 0.9" in m.value for m in app.markdown)
    assert any("Unsupported number: 0.8" in m.value for m in app.markdown)


ACCEPTED_TARGET = "src.app.accepted_answers.save_accepted_answer"


def accept_button(at: AppTest, index: int):
    return at.button(key=f"accept_button_{index}")


def test_accept_button_saves_question_and_that_versions_answer(app):
    with patch(SERVICE_TARGET, return_value=approved_result()), patch(
        ACCEPTED_TARGET, return_value=7
    ) as save:
        fill_and_run(app)
        accept_button(app, 0).click()
        app.run()

    assert not app.exception
    save.assert_called_once_with(
        "What was the hazard ratio?",
        "The hazard ratio was 0.75 [Source: trial.pdf, p. 2].",
    )
    assert any("Accepted answer saved (id 7)." in s.value for s in app.success)
    assert not any(b.key == "accept_button_0" for b in app.button)


def test_each_version_has_its_own_accept_button(app):
    with patch(SERVICE_TARGET, return_value=rejected_result()), patch(
        ACCEPTED_TARGET, return_value=3
    ) as save:
        fill_and_run(app)
        accept_button(app, 1).click()
        app.run()

    assert not app.exception
    save.assert_called_once_with("What was the hazard ratio?", "The hazard ratio was 0.8.")
    assert any(b.key == "accept_button_0" for b in app.button)
    assert not any(b.key == "accept_button_1" for b in app.button)


def test_accept_button_is_available_for_unapproved_versions(app):
    with patch(SERVICE_TARGET, return_value=rejected_result()):
        fill_and_run(app)

    assert not app.exception
    assert accept_button(app, 0) is not None
    assert accept_button(app, 1) is not None


def test_accept_failure_is_shown_and_button_stays(app):
    with patch(SERVICE_TARGET, return_value=approved_result()), patch(
        ACCEPTED_TARGET, side_effect=OSError("disk full")
    ):
        fill_and_run(app)
        accept_button(app, 0).click()
        app.run()

    assert not app.exception
    assert any("Could not save the accepted answer: disk full" in e.value for e in app.error)
    assert accept_button(app, 0) is not None


LOAD_TARGET = "src.app.accepted_answers.load_accepted_answers"
SAVED_ROWS = [
    {
        "id": 2,
        "created_at": "2026-10-10 19:40:00",
        "question": "Second question about placebo?",
        "answer": "Second accepted answer [Source: trial.pdf, p. 2].",
    },
    {
        "id": 1,
        "created_at": "2026-10-10 19:36:00",
        "question": "First question?",
        "answer": "First accepted answer.",
    },
]


def test_accepted_answers_tab_shows_saved_rows():
    with patch(LOAD_TARGET, return_value=SAVED_ROWS) as load:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

    assert not at.exception
    load.assert_called_with("")
    assert any(t.label == "Accepted Answers" for t in at.tabs)
    markdown = " ".join(m.value for m in at.markdown)
    assert "**Question:** Second question about placebo?" in markdown
    assert "Second accepted answer" in markdown
    assert "**Question:** First question?" in markdown
    assert any("2 accepted answer(s)" in c.value for c in at.caption)


def test_accepted_answers_tab_shows_empty_message():
    with patch(LOAD_TARGET, return_value=[]):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

    assert not at.exception
    assert any("No answers have been accepted yet." in i.value for i in at.info)


def test_accepted_answers_filter_passes_keyword_to_loader():
    with patch(LOAD_TARGET, return_value=SAVED_ROWS) as load:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.text_input(key="accepted_filter").input("placebo")
        at.run()

    assert not at.exception
    assert load.call_args.args[0] == "placebo"


def test_accepted_answers_filter_with_no_match_shows_message():
    with patch(LOAD_TARGET, return_value=[]):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.text_input(key="accepted_filter").input("nothing matches this")
        at.run()

    assert not at.exception
    assert any("No accepted answers match." in i.value for i in at.info)


CHUNK_LOAD_TARGET = "src.app.chunk_browser.load_chunks"
CHUNK_SOURCES_TARGET = "src.app.chunk_browser.list_chunk_sources"
CHUNK_ROWS = [
    {
        "id": "a-1",
        "source_file": "trial.pdf",
        "page_number": 1,
        "chunk_type": "table",
        "section_header": "Baseline",
        "content": "Placebo group had 42 patients.",
    },
]


def test_chunk_table_tab_shows_matching_chunks():
    with patch(CHUNK_SOURCES_TARGET, return_value=["trial.pdf"]), patch(
        CHUNK_LOAD_TARGET, return_value=(CHUNK_ROWS, 1)
    ) as load:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

    assert not at.exception
    load.assert_called_with("", "")
    assert any(t.label == "Chunk Table" for t in at.tabs)
    assert any("1 matching chunk(s)." in c.value for c in at.caption)


def test_chunk_table_shows_preview_note_when_truncated():
    with patch(CHUNK_SOURCES_TARGET, return_value=["trial.pdf"]), patch(
        CHUNK_LOAD_TARGET, return_value=(CHUNK_ROWS, 250)
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

    assert not at.exception
    assert any("250 matching chunk(s). Showing the first 1." in c.value for c in at.caption)


def test_chunk_table_empty_database_message():
    with patch(CHUNK_SOURCES_TARGET, return_value=[]), patch(CHUNK_LOAD_TARGET, return_value=([], 0)):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

    assert not at.exception
    assert any("The chunk table is empty or missing." in i.value for i in at.info)


def test_chunk_table_keyword_filter_passes_keyword_to_loader():
    with patch(CHUNK_SOURCES_TARGET, return_value=["trial.pdf"]), patch(
        CHUNK_LOAD_TARGET, return_value=(CHUNK_ROWS, 1)
    ) as load:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.text_input(key="chunk_filter").input("placebo")
        at.run()

    assert not at.exception
    assert load.call_args.args[0] == "placebo"


def test_chunk_table_source_selector_passes_source_to_loader():
    with patch(CHUNK_SOURCES_TARGET, return_value=["dosing.pdf", "trial.pdf"]), patch(
        CHUNK_LOAD_TARGET, return_value=(CHUNK_ROWS, 1)
    ) as load:
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.selectbox(key="chunk_source").select("trial.pdf")
        at.run()

    assert not at.exception
    assert load.call_args.args[1] == "trial.pdf"


def test_chunk_table_shows_warning_when_database_is_locked():
    locked = duckdb.IOException("File is already open in another process")
    with patch(CHUNK_SOURCES_TARGET, return_value=[]), patch(CHUNK_LOAD_TARGET, side_effect=locked):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

    assert not at.exception
    assert any("Could not read the chunk table" in w.value for w in at.warning)
    assert any(t.label == "Ask & Verify" for t in at.tabs)
    assert any(t.label == "Accepted Answers" for t in at.tabs)


def test_chunk_table_no_match_message():
    with patch(CHUNK_SOURCES_TARGET, return_value=["trial.pdf"]), patch(
        CHUNK_LOAD_TARGET, return_value=([], 0)
    ):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.text_input(key="chunk_filter").input("nothing matches this")
        at.run()

    assert not at.exception
    assert any("No chunks match the filters." in i.value for i in at.info)

