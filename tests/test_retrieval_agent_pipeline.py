"""
Tests for the retrieval and agent pipeline: revision loop, limits, fail-fast
credentials, and CLI exit codes. The LLM and retriever are fakes.
"""

from typing import List
from unittest.mock import patch

import pytest

from src.retrieval_agent import run as run_module
from src.retrieval_agent.bedrock_client import BedrockError
from src.retrieval_agent.fact_check import FACT_CHECK_SYSTEM_PROMPT
from src.retrieval_agent.retrieval import RetrievalError, RetrievalResult, RetrievedChunk
from src.retrieval_agent.run import main, run_pipeline
from src.retrieval_agent.synthesis import SYNTHESIS_SYSTEM_PROMPT
from src.utils.config import Config

APPROVED_JSON = '{"is_approved": true, "critique": "", "unsupported_items": []}'
BAD_DRAFT = "The response rate was 99 [Source: trial.pdf, p. 2]."
GOOD_DRAFT = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."

CHUNK = RetrievedChunk(
    chunk_id="c1",
    source_file="trial.pdf",
    page_number=2,
    chunk_type="text",
    section_header="Results",
    content="The hazard ratio was 0.75 across the cohort.",
    similarity_score=0.92,
)


class ScriptedLLM:
    """Returns queued drafts for synthesis calls and approval JSON for fact-checks."""

    def __init__(self, drafts: List[str]):
        self.drafts = list(drafts)
        self.synthesis_prompts: List[str] = []
        self.fact_check_calls = 0

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        if system_prompt == FACT_CHECK_SYSTEM_PROMPT:
            self.fact_check_calls += 1
            return APPROVED_JSON
        assert system_prompt == SYNTHESIS_SYSTEM_PROMPT
        self.synthesis_prompts.append(user_prompt)
        return self.drafts.pop(0)


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = [CHUNK] if chunks is None else chunks
        self.calls: List[str] = []

    def retrieve(self, question: str, top_k=None) -> RetrievalResult:
        self.calls.append(question)
        return RetrievalResult(
            question=question,
            route="both",
            query_dimensions=1024,
            chunks=list(self.chunks),
        )

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


class TestRevisionLoop:
    def test_revises_failed_draft_until_approved(self):
        llm = ScriptedLLM([BAD_DRAFT, GOOD_DRAFT])
        result = run_pipeline("What was the hazard ratio?", llm=llm, retriever=FakeRetriever(), max_revisions=2)

        assert result.is_approved is True
        assert result.revisions == 1
        assert result.draft == GOOD_DRAFT
        assert result.final_answer == GOOD_DRAFT
        assert llm.fact_check_calls == 2

    def test_revision_prompt_contains_critique(self):
        llm = ScriptedLLM([BAD_DRAFT, GOOD_DRAFT])
        run_pipeline("What was the hazard ratio?", llm=llm, retriever=FakeRetriever(), max_revisions=2)

        assert "Previous draft:" in llm.synthesis_prompts[1]
        assert BAD_DRAFT in llm.synthesis_prompts[1]
        assert "Unsupported number: 99" in llm.synthesis_prompts[1]

    def test_approved_first_draft_needs_no_revision(self):
        llm = ScriptedLLM([GOOD_DRAFT])
        result = run_pipeline("What was the hazard ratio?", llm=llm, retriever=FakeRetriever())

        assert result.revisions == 0
        assert result.final_answer == GOOD_DRAFT
        assert len(llm.synthesis_prompts) == 1

    def test_exhausted_revisions_do_not_publish_answer(self):
        llm = ScriptedLLM([BAD_DRAFT, BAD_DRAFT, BAD_DRAFT])
        result = run_pipeline("What was the response rate?", llm=llm, retriever=FakeRetriever(), max_revisions=2)

        assert result.is_approved is False
        assert result.revisions == 2
        assert result.final_answer is None
        assert result.draft == BAD_DRAFT
        assert len(llm.synthesis_prompts) == 3

    def test_zero_max_revisions_stops_after_first_check(self):
        llm = ScriptedLLM([BAD_DRAFT])
        result = run_pipeline("What was the response rate?", llm=llm, retriever=FakeRetriever(), max_revisions=0)

        assert result.revisions == 0
        assert result.is_approved is False
        assert len(llm.synthesis_prompts) == 1

    def test_negative_max_revisions_rejected(self):
        with pytest.raises(ValueError, match="max_revisions"):
            run_pipeline("q", llm=ScriptedLLM([GOOD_DRAFT]), retriever=FakeRetriever(), max_revisions=-1)


class TestAttempts:
    def test_every_version_is_recorded_with_its_own_check(self):
        llm = ScriptedLLM([BAD_DRAFT, GOOD_DRAFT])
        result = run_pipeline("What was the hazard ratio?", llm=llm, retriever=FakeRetriever(), max_revisions=2)

        assert [a.label for a in result.attempts] == ["Initial draft", "Revision 1"]
        assert [a.draft for a in result.attempts] == [BAD_DRAFT, GOOD_DRAFT]
        assert result.attempts[0].fact_check.is_approved is False
        assert result.attempts[0].fact_check.issues
        assert result.attempts[1].fact_check.is_approved is True

    def test_last_attempt_matches_top_level_fields(self):
        llm = ScriptedLLM([BAD_DRAFT, BAD_DRAFT, BAD_DRAFT])
        result = run_pipeline("What was the response rate?", llm=llm, retriever=FakeRetriever(), max_revisions=2)

        assert [a.label for a in result.attempts] == ["Initial draft", "Revision 1", "Revision 2"]
        assert result.attempts[-1].draft == result.draft
        assert result.attempts[-1].fact_check is result.fact_check

    def test_approved_first_draft_has_single_attempt(self):
        llm = ScriptedLLM([GOOD_DRAFT])
        result = run_pipeline("What was the hazard ratio?", llm=llm, retriever=FakeRetriever())

        assert [a.label for a in result.attempts] == ["Initial draft"]

    def test_no_chunks_records_no_attempts(self):
        result = run_pipeline("q", llm=ScriptedLLM([]), retriever=FakeRetriever(chunks=[]))

        assert result.attempts == []


class TestNoEvidence:
    def test_no_chunks_skips_llm_entirely(self):
        llm = ScriptedLLM([])
        result = run_pipeline("What was the hazard ratio?", llm=llm, retriever=FakeRetriever(chunks=[]))

        assert result.is_approved is False
        assert result.final_answer is None
        assert result.draft is None
        assert llm.synthesis_prompts == []
        assert llm.fact_check_calls == 0
        assert any("No chunks" in line for line in result.logs)


class TestFailFast:
    def test_missing_credentials_fail_before_retrieval(self):
        retriever = FakeRetriever()
        with pytest.raises(BedrockError):
            run_pipeline(
                "What was the hazard ratio?",
                aws_access_key_id="",
                aws_secret_access_key="",
                retriever=retriever,
            )
        assert retriever.calls == []

    def test_retrieval_error_propagates(self):
        class BrokenRetriever(FakeRetriever):
            def retrieve(self, question, top_k=None):
                raise RetrievalError("boom")

        with pytest.raises(RetrievalError, match="boom"):
            run_pipeline("q", llm=ScriptedLLM([]), retriever=BrokenRetriever())


class TestCli:
    def test_retrieve_only_prints_chunks_and_returns_zero(self, capsys):
        with patch.object(run_module, "Retriever") as retriever_cls:
            retriever_cls.return_value = FakeRetriever()
            exit_code = main(["What was the hazard ratio?", "--retrieve-only"])

        assert exit_code == 0
        output = capsys.readouterr().out
        assert "trial.pdf" in output
        assert "p. 2" in output
        assert "dimensions: 1024" in output

    def test_full_mode_without_credentials_returns_two(self, monkeypatch, capsys):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        with patch.object(run_module, "Retriever") as retriever_cls:
            retriever_cls.return_value = FakeRetriever()
            exit_code = main(["What was the hazard ratio?"])

        assert exit_code == 2
        assert "Error" in capsys.readouterr().err

    def test_full_mode_exit_code_reflects_approval(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        llm = ScriptedLLM([GOOD_DRAFT])
        with patch.object(run_module.BedrockLLM, "from_credentials", return_value=llm), \
                patch.object(run_module, "Retriever") as retriever_cls:
            retriever_cls.return_value = FakeRetriever()
            assert main(["What was the hazard ratio?"]) == 0

        llm_bad = ScriptedLLM([BAD_DRAFT])
        with patch.object(run_module.BedrockLLM, "from_credentials", return_value=llm_bad), \
                patch.object(run_module, "Retriever") as retriever_cls:
            retriever_cls.return_value = FakeRetriever()
            assert main(["What was the response rate?", "--max-revisions", "0"]) == 1
