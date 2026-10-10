"""
Tests for citation filename canonicalization in the synthesis agent.
"""

from typing import List

from src.retrieval_agent.fact_check import FactChecker
from src.retrieval_agent.retrieval import RetrievedChunk
from src.retrieval_agent.synthesis import SynthesisAgent, canonicalize_citations

APPROVED_JSON = '{"is_approved": true, "critique": "", "unsupported_items": []}'


class ScriptedLLM:
    """Returns a fixed reply for every call."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: List[tuple] = []

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


def make_chunk(source: str, page: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source}-{page}",
        source_file=source,
        page_number=page,
        chunk_type="text",
        section_header="Results",
        content=content,
        similarity_score=0.9,
    )


CHUNKS = [
    make_chunk("Clinical_study_designs.pdf", 6, "Parallel-group trials randomize patients."),
    make_chunk("Clinical-Trials.pdf", 2, "Phase 3 trials enroll large cohorts."),
]


class TestCanonicalizeCitations:
    def test_hyphenated_name_is_rewritten_to_stored_name(self):
        draft = "Parallel-group trials randomize patients. [Source: Clinical-study-designs.pdf, p. 6]"

        result = canonicalize_citations(draft, CHUNKS)

        assert result == "Parallel-group trials randomize patients. [Source: Clinical_study_designs.pdf, p. 6]"

    def test_case_difference_is_rewritten(self):
        draft = "Phase 3 trials enroll large cohorts. [Source: clinical-trials.PDF, p. 2]"

        result = canonicalize_citations(draft, CHUNKS)

        assert result == "Phase 3 trials enroll large cohorts. [Source: Clinical-Trials.pdf, p. 2]"

    def test_exact_name_is_unchanged(self):
        draft = "Text. [Source: Clinical_study_designs.pdf, p. 6]"

        assert canonicalize_citations(draft, CHUNKS) == draft

    def test_unknown_source_is_left_unchanged(self):
        draft = "Text. [Source: Other-study.pdf, p. 6]"

        assert canonicalize_citations(draft, CHUNKS) == draft

    def test_ambiguous_match_is_left_unchanged(self):
        chunks = [
            make_chunk("Clinical-Trials.pdf", 1, "a"),
            make_chunk("Clinical_Trials.pdf", 1, "b"),
        ]
        draft = "Text. [Source: clinicaltrials.pdf, p. 1]"

        assert canonicalize_citations(draft, chunks) == draft

    def test_rewritten_draft_passes_unknown_source_check(self):
        draft = "Parallel-group trials randomize patients. [Source: Clinical-study-designs.pdf, p. 6]"
        checker = FactChecker(ScriptedLLM(APPROVED_JSON))

        before = checker.check(draft, CHUNKS)
        after = checker.check(canonicalize_citations(draft, CHUNKS), CHUNKS)

        assert any("Unknown source cited" in issue for issue in before.issues)
        assert not any("Unknown source cited" in issue for issue in after.issues)


class TestSynthesisAgentCanonicalizes:
    def test_draft_uses_stored_filename(self):
        llm = ScriptedLLM("Parallel-group trials randomize patients. [Source: Clinical-study-designs.pdf, p. 6]")

        draft = SynthesisAgent(llm).draft("What is a parallel design?", CHUNKS)

        assert "[Source: Clinical_study_designs.pdf, p. 6]" in draft

    def test_revision_uses_stored_filename(self):
        llm = ScriptedLLM("Parallel-group trials randomize patients. [Source: Clinical-study-designs.pdf, p. 6]")

        revised = SynthesisAgent(llm).revise("q", CHUNKS, "old draft", "Uncited claim: x")

        assert "[Source: Clinical_study_designs.pdf, p. 6]" in revised
