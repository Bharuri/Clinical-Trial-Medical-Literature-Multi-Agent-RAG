"""
Tests for the fact-check agent: deterministic citation/number rules and the LLM verdict.
"""

from typing import List

from src.retrieval_agent.fact_check import FactChecker
from src.retrieval_agent.retrieval import RetrievedChunk

APPROVED_JSON = '{"is_approved": true, "critique": "", "unsupported_items": []}'
REJECTED_JSON = '{"is_approved": false, "critique": "Remove the unsupported claim.", "unsupported_items": ["claim"]}'


class FakeLLM:
    """Returns a scripted verdict and records the prompts it received."""

    def __init__(self, response: str = APPROVED_JSON):
        self.response = response
        self.calls: List[tuple] = []

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def make_chunk(source: str, page: int, content: str, chunk_type: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{source}-{page}",
        source_file=source,
        page_number=page,
        chunk_type=chunk_type,
        section_header="Results",
        content=content,
        similarity_score=0.9,
    )


CHUNKS = [
    make_chunk("trial.pdf", 2, "The hazard ratio was 0.75 (95% CI 0.6 to 0.9) across 1234 patients."),
    make_chunk("trial.pdf", 3, "A total of 42 patients discontinued treatment."),
]


def check(draft: str, llm: FakeLLM = None, chunks=None):
    return FactChecker(llm or FakeLLM()).check(draft, chunks if chunks is not None else CHUNKS)


class TestApproval:
    def test_fully_supported_draft_is_approved(self):
        draft = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
        result = check(draft)

        assert result.is_approved is True
        assert result.issues == []
        assert result.critique == ""

    def test_thousands_separator_matches_source_number(self):
        draft = "The study enrolled 1,234 patients [Source: trial.pdf, p. 2]."
        assert check(draft).is_approved is True

    def test_table_rows_and_headings_are_skipped(self):
        draft = (
            "# Summary\n"
            "| Arm | Count |\n"
            "|---|---|\n"
            "| A | 99 |\n"
            "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
        )
        assert check(draft).issues == []

    def test_citation_after_period_stays_with_sentence(self):
        draft = "The discontinuation count was 42. [Source: trial.pdf, p. 3]"
        assert check(draft).issues == []

    def test_one_citation_covers_preceding_sentences_in_bullet(self):
        draft = (
            "- Patients are randomized to the new or standard treatment. "
            "They are followed up in parallel groups. [Source: trial.pdf, p. 2]"
        )
        result = check(draft)

        assert result.issues == []
        assert result.is_approved is True

    def test_number_in_grouped_sentence_is_checked_against_citation_page(self):
        draft = "The hazard ratio was 0.75 across the trial. [Source: trial.pdf, p. 3]"
        result = check(draft)

        assert result.is_approved is False
        assert result.issues

    def test_llm_rejection_blocks_approval_even_with_clean_rules(self):
        draft = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
        result = check(draft, FakeLLM(REJECTED_JSON))

        assert result.is_approved is False
        assert result.issues == []
        assert "Remove the unsupported claim." in result.critique
        assert "claim" in result.critique

    def test_unparseable_llm_response_is_not_approved(self):
        draft = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
        result = check(draft, FakeLLM("I think it looks fine"))

        assert result.is_approved is False
        assert "could not be parsed" in result.critique

    def test_llm_receives_sources_and_draft(self):
        llm = FakeLLM()
        draft = "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
        check(draft, llm)

        assert len(llm.calls) == 1
        _, user_prompt = llm.calls[0]
        assert draft in user_prompt
        assert "source_file: trial.pdf | page_number: 2" in user_prompt

    def test_source_headers_have_no_list_index(self):
        from src.retrieval_agent.synthesis import format_sources

        rendered = format_sources(CHUNKS)

        assert rendered.startswith("source_file: trial.pdf | page_number: 2")
        assert "[1]" not in rendered
        assert "[2]" not in rendered

    def test_index_as_source_is_flagged_as_unknown(self):
        draft = "The hazard ratio was 0.75 [Source: 1, p. 2]."
        result = check(draft)

        assert result.is_approved is False
        assert any("Unknown source cited: 1, p. 2" in issue for issue in result.issues)


class TestCitationRules:
    def test_wrong_page_flagged(self):
        draft = "The hazard ratio was 0.75 [Source: trial.pdf, p. 9]."
        result = check(draft)

        assert result.is_approved is False
        assert any("Page not retrieved" in issue and "p. 9" in issue for issue in result.issues)

    def test_unknown_source_flagged(self):
        draft = "The hazard ratio was 0.75 [Source: other.pdf, p. 2]."
        result = check(draft)

        assert result.is_approved is False
        assert any("Unknown source" in issue for issue in result.issues)

    def test_uncited_claim_flagged(self):
        draft = "The drug is effective in all patients."
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_short_uncited_fragment_not_flagged(self):
        draft = "Yes."
        assert check(draft).issues == []

    def test_sentence_after_last_citation_in_bullet_is_covered_by_block(self):
        draft = (
            "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]. "
            "The drug is effective in all patients."
        )
        result = check(draft)

        assert result.issues == []

    def test_uncited_sentence_in_cited_block_is_caught_by_llm_check(self):
        draft = (
            "LLMs can assist in systematic literature reviews and trial design/optimization. "
            "The hazard ratio was 0.75 [Source: trial.pdf, p. 2]."
        )
        result = check(draft, FakeLLM(REJECTED_JSON))

        assert result.issues == []
        assert result.is_approved is False
        assert "Remove the unsupported claim." in result.critique

    def test_uncited_block_on_own_line_flagged(self):
        draft = (
            "The hazard ratio was 0.75 [Source: trial.pdf, p. 2].\n"
            "LLMs can assist in systematic literature reviews and trial design/optimization."
        )
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_canonical_refusal_not_flagged(self):
        draft = "The provided sources do not contain information to answer this question."
        result = check(draft)

        assert result.issues == []
        assert result.is_approved is True

    def test_refusal_wording_variant_not_flagged(self):
        draft = "The sources do not mention the dosing guideline."
        assert check(draft).issues == []

    def test_refusal_with_claim_still_flagged(self):
        draft = (
            "The sources do not mention the dosing schedule, but the drug is clearly effective "
            "in all patients."
        )
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_refusal_with_number_still_flagged(self):
        draft = "The sources do not contain the 2030 target of 75 patients."
        result = check(draft)

        assert any("Uncited claim" in issue for issue in result.issues)


class TestBlockCoverage:
    def test_lead_in_line_ending_with_colon_is_not_fact_checked(self):
        draft = (
            "Researchers can utilize the following treatment control variations when designing a trial:\n"
            "- Active treatment concurrent control compares a new drug to a standard drug. "
            "[Source: trial.pdf, p. 2]"
        )
        result = check(draft)

        assert result.issues == []
        assert result.is_approved is True

    def test_lead_in_exemption_applies_only_to_first_line(self):
        draft = (
            "- Active treatment concurrent control compares a new drug to a standard drug. "
            "[Source: trial.pdf, p. 2]\n"
            "Researchers can utilize the following treatment control variations when designing a trial:"
        )
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_first_line_without_colon_is_still_fact_checked(self):
        draft = "Researchers can utilize the following treatment control variations in a trial."
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    @staticmethod
    def _bullets(cited: int, uncited: int) -> str:
        labels = [chr(ord("A") + i) for i in range(cited + uncited)]
        lines = [f"- Finding {label} was reported in the trial. [Source: trial.pdf, p. 2]" for label in labels[:cited]]
        lines += [f"- Finding {label} was reported in the trial without a source." for label in labels[cited:]]
        return "\n".join(lines)

    def test_nine_of_ten_blocks_cited_passes_rule_check(self):
        result = check(self._bullets(cited=9, uncited=1))

        assert result.issues == []
        assert "Citation coverage: 9/10" in " ".join(result.logs)

    def test_eight_of_ten_blocks_cited_fails_and_lists_uncited_blocks(self):
        result = check(self._bullets(cited=8, uncited=2))

        assert result.is_approved is False
        uncited = [issue for issue in result.issues if issue.startswith("Uncited claim")]
        assert len(uncited) == 2

    def test_citation_error_fails_even_when_coverage_passes(self):
        draft = self._bullets(cited=9, uncited=0) + "\n- The response rate was 99 [Source: trial.pdf, p. 2]."
        result = check(draft)

        assert result.is_approved is False
        assert any("Unsupported number: 99" in issue for issue in result.issues)


class TestNonClaimBlocks:
    def test_navigation_sentence_without_citation_not_flagged(self):
        result = check("Please see the explanation below for the details.")

        assert result.issues == []
        assert result.is_approved is True

    def test_restatement_of_cited_text_not_flagged(self):
        draft = (
            "Patients were randomized to the new treatment or standard treatment "
            "[Source: trial.pdf, p. 2].\n"
            "This means patients were randomized to the new or standard treatment."
        )
        result = check(draft)

        assert result.issues == []
        assert result.is_approved is True

    def test_new_uncited_claim_with_number_still_flagged(self):
        draft = (
            "The hazard ratio was 0.75 [Source: trial.pdf, p. 2].\n"
            "The response rate was 45 percent in the active arm."
        )
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_uncited_claim_with_little_overlap_still_flagged(self):
        draft = (
            "Patients were randomized to the new treatment or standard treatment "
            "[Source: trial.pdf, p. 2].\n"
            "The drug is effective in all patients."
        )
        result = check(draft)

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_signpost_with_claim_verb_still_flagged(self):
        result = check("The table below shows the drug is effective in all patients.")

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_conclusion_claim_still_flagged(self):
        result = check("In summary, the drug is effective in all patients.")

        assert result.is_approved is False
        assert any("Uncited claim" in issue for issue in result.issues)

    def test_navigation_sentence_is_excluded_from_coverage(self):
        draft = TestBlockCoverage._bullets(cited=9, uncited=1) + "\nSee the explanation below."
        result = check(draft)

        assert result.issues == []
        assert "Citation coverage: 9/10" in " ".join(result.logs)
        assert "Non-claim blocks skipped" in " ".join(result.logs)


class TestNumberRules:
    def test_unsupported_number_flagged(self):
        draft = "The response rate was 99 [Source: trial.pdf, p. 2]."
        result = check(draft)

        assert result.is_approved is False
        assert any("Unsupported number: 99" in issue for issue in result.issues)

    def test_number_on_other_retrieved_page_is_citation_mismatch(self):
        draft = "The discontinuation count was 42 [Source: trial.pdf, p. 2]."
        result = check(draft)

        assert result.is_approved is False
        assert any("Citation mismatch: 42" in issue for issue in result.issues)

    def test_number_citation_on_correct_page_passes(self):
        draft = "The discontinuation count was 42 [Source: trial.pdf, p. 3]."
        assert check(draft).is_approved is True
