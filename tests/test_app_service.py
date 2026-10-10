"""
Tests for the Streamlit service layer. The embedding model, DuckDB, and
Bedrock are replaced with fakes, so no model is loaded and no AWS call is made.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.app import service
from src.retrieval_agent.bedrock_client import BedrockError
from src.retrieval_agent.fact_check import FACT_CHECK_SYSTEM_PROMPT
from src.retrieval_agent.retrieval import RetrievalResult, RetrievedChunk
from src.retrieval_agent.run import PipelineResult
from src.retrieval_agent.synthesis import SYNTHESIS_SYSTEM_PROMPT
from src.utils.config import Config

CREDS = dict(
    aws_access_key_id="AKIATEST",
    aws_secret_access_key="secret",
    aws_region_name="us-east-1",
    model_id="test-model",
)

CHUNK = RetrievedChunk(
    chunk_id="c1",
    source_file="trial.pdf",
    page_number=2,
    chunk_type="text",
    section_header="Results",
    content="The hazard ratio was 0.75 across the cohort.",
    similarity_score=0.92,
)

GOOD_DRAFT = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."


class ScriptedLLM:
    """Returns a fixed draft for synthesis and an approval verdict for fact-checks."""

    def __init__(self, draft: str):
        self.draft = draft

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        if system_prompt == FACT_CHECK_SYSTEM_PROMPT:
            return '{"is_approved": true, "critique": "", "unsupported_items": []}'
        assert system_prompt == SYNTHESIS_SYSTEM_PROMPT
        return self.draft


class FakeRetriever:
    def __init__(self):
        self.closed = False

    def retrieve(self, question: str, top_k=None) -> RetrievalResult:
        return RetrievalResult(question=question, route="text", query_dimensions=768, chunks=[CHUNK])

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


class TestAnswerQuestion:
    def test_empty_question_rejected_before_any_loading(self):
        encoder_factory = MagicMock()
        with patch.object(service, "get_query_encoder", encoder_factory):
            with pytest.raises(ValueError, match="research question"):
                service.answer_question("   ", **CREDS)

        encoder_factory.assert_not_called()

    def test_missing_credentials_fail_before_model_load(self):
        encoder_factory = MagicMock()
        with patch.object(service, "get_query_encoder", encoder_factory):
            with pytest.raises(BedrockError):
                service.answer_question(
                    "What was the hazard ratio?",
                    **{**CREDS, "aws_secret_access_key": ""},
                )

        encoder_factory.assert_not_called()

    def test_owned_retriever_is_built_with_cached_encoder_and_closed(self):
        encoder = object()
        retriever_cls = MagicMock()
        sentinel = PipelineResult(question="q")
        with patch.object(service, "get_query_encoder", return_value=encoder), \
                patch.object(service, "Retriever", retriever_cls), \
                patch.object(service, "run_pipeline", return_value=sentinel) as run:
            result = service.answer_question("q", **CREDS)

        retriever_cls.assert_called_once_with(encoder=encoder)
        built = retriever_cls.return_value
        built.close.assert_called_once()
        assert run.call_args.kwargs["retriever"] is built
        assert run.call_args.kwargs["llm"] is not None
        assert result is sentinel

    def test_owned_retriever_closed_even_when_pipeline_fails(self):
        retriever_cls = MagicMock()
        with patch.object(service, "get_query_encoder", return_value=object()), \
                patch.object(service, "Retriever", retriever_cls), \
                patch.object(service, "run_pipeline", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                service.answer_question("q", **CREDS)

        retriever_cls.return_value.close.assert_called_once()

    def test_injected_retriever_and_llm_are_not_closed(self):
        retriever = MagicMock()
        llm = object()
        with patch.object(service, "run_pipeline", return_value=PipelineResult(question="q")) as run:
            service.answer_question("q", llm=llm, retriever=retriever, **CREDS)

        retriever.close.assert_not_called()
        assert run.call_args.kwargs["llm"] is llm

    def test_end_to_end_approved_answer_with_fakes(self):
        result = service.answer_question(
            "What was the hazard ratio?",
            llm=ScriptedLLM(GOOD_DRAFT),
            retriever=FakeRetriever(),
            **CREDS,
        )

        assert result.is_approved
        assert result.final_answer == GOOD_DRAFT
        assert result.retrieval.match_count == 1
        assert result.retrieval.query_dimensions == 768
