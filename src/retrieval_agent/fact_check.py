"""
Fact-check agent: verifies that a cited draft is supported by the retrieved
chunks. Deterministic rules run first; an LLM support check runs after them.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Set, Tuple

from src.retrieval_agent.bedrock_client import TextModel
from src.retrieval_agent.retrieval import RetrievedChunk
from src.retrieval_agent.synthesis import CITATION_PATTERN, format_sources

_NUMBER_PATTERN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_MIN_UNCITED_WORDS = 4
# Share of factual bullets or paragraphs that must carry a citation.
_MIN_CITED_BLOCK_RATIO = 0.9
# Short statements that the sources lack the answer are not claims and need no citation.
_MAX_NOT_FOUND_WORDS = 15
_NOT_FOUND_PATTERN = re.compile(
    r"\b(?:sources?|documents?|excerpts?|chunks?|text)\b[^.]{0,40}?\b(?:do|does)\s+not\s+"
    r"(?:contain|provide|mention|include|address|state|specify)\b"
    r"|\bno\s+(?:information|data|evidence)\b[^.]{0,40}?\bin\s+the\s+(?:provided\s+)?"
    r"(?:sources?|documents?|text)\b",
    re.IGNORECASE,
)
# Navigation sentences such as "see the explanation below" make no claim of their own.
# A sentence with a claim verb is not navigation, e.g. "The table below shows the drug is effective."
_MAX_SIGNPOST_WORDS = 30
_SIGNPOST_PATTERN = re.compile(
    r"\b(?:see|refer to|shown|described|outlined|summari[sz]ed|listed|presented|detailed)\b[^.]{0,40}?"
    r"\b(?:below|above|following)\b"
    r"|\b(?:explanation|details|table|figure|section|list|summary)\b[^.]{0,20}?\b(?:below|above|follows)\b",
    re.IGNORECASE,
)
_CLAIM_VERB_PATTERN = re.compile(
    r"\b(?:is|are|was|were|be|been|shows?|showed|found|reports?|reported|demonstrat\w*|"
    r"increas\w*|decreas\w*|reduc\w*|improv\w*|associated|compared|significant\w*|effective)\b",
    re.IGNORECASE,
)
# An uncited sentence whose content words mostly appear in earlier cited text only explains that text.
_MIN_RESTATED_OVERLAP = 0.8
_STOPWORDS = frozenset(
    "the and for are was were with that this from have has had its their there these those "
    "which who not but can may also into than then such each both been being will would "
    "should could about within between over under across through using use used one two".split()
)

FACT_CHECK_SYSTEM_PROMPT = """You are the fact-checker for a clinical trial and medical literature assistant.
Verify that every claim in the draft answer is directly supported by the provided sources.

Check:
- Each factual claim, metric, and number is stated in the sources.
- Each table value matches the source row and column.
- The answer does not add outside knowledge or overstate the sources.
- Skip an opening line that only introduces a list and ends with a colon.
- Skip navigation sentences such as "see the explanation below", and sentences that only explain or restate a claim already cited in the draft. These need no citation.

Respond with ONLY a JSON object, no other text:
{"is_approved": true or false, "critique": "specific fixes needed, or empty string", "unsupported_items": ["item", ...]}"""


@dataclass
class FactCheckResult:
    is_approved: bool
    critique: str = ""
    issues: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _normalize_number(token: str) -> float:
    return float(token.replace(",", ""))


def _extract_numbers(text: str) -> Set[float]:
    return {_normalize_number(m) for m in _NUMBER_PATTERN.findall(text)}


def _split_blocks(draft: str) -> List[str]:
    """Return the bullet or paragraph lines of a draft, skipping headings, tables, and rules."""
    blocks: List[str] = []
    for raw_line in draft.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "|")) or re.fullmatch(r"[-:\s]+", line):
            continue
        blocks.append(re.sub(r"^([-*+]|\d+\.)\s+", "", line))
    return blocks


def _content_words(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _is_signpost(body: str, has_numbers: bool, body_words: int) -> bool:
    return (
        not has_numbers
        and body_words <= _MAX_SIGNPOST_WORDS
        and bool(_SIGNPOST_PATTERN.search(body))
        and not _CLAIM_VERB_PATTERN.search(body)
    )


def _restates_cited_text(body: str, has_numbers: bool, cited_words: Set[str]) -> bool:
    if has_numbers:
        return False
    words = _content_words(body)
    if not words or not cited_words:
        return False
    return len(words & cited_words) / len(words) >= _MIN_RESTATED_OVERLAP


class FactChecker:
    """Checks a cited draft against retrieved chunks."""

    def __init__(self, llm: TextModel):
        self.llm = llm

    def check(self, draft: str, chunks: List[RetrievedChunk]) -> FactCheckResult:
        logs: List[str] = []
        issues = self._rule_issues(draft, chunks, logs)
        llm_approved, llm_critique = self._llm_check(draft, chunks, logs)

        critique_parts = list(issues)
        if llm_critique:
            critique_parts.append(llm_critique)
        critique = "\n".join(critique_parts)

        is_approved = not issues and llm_approved
        logs.append(f"Fact-check result: approved={is_approved}, rule_issues={len(issues)}")
        return FactCheckResult(
            is_approved=is_approved,
            critique=critique if not is_approved else "",
            issues=issues,
            logs=logs,
        )

    def _rule_issues(
        self, draft: str, chunks: List[RetrievedChunk], logs: List[str]
    ) -> List[str]:
        issues: List[str] = []
        uncited_blocks: List[str] = []
        factual_blocks = 0
        cited_blocks = 0
        pages_by_source: Dict[str, Set[int]] = {}
        for chunk in chunks:
            pages_by_source.setdefault(chunk.source_file, set()).add(chunk.page_number)

        blocks = _split_blocks(draft)
        cited_words: Set[str] = set()
        non_claim_blocks = 0
        for index, block in enumerate(blocks):
            citations = CITATION_PATTERN.findall(block)
            # Opening line that introduces a list is not a factual claim.
            if index == 0 and not citations and block.endswith(":"):
                non_claim_blocks += 1
                continue
            body = CITATION_PATTERN.sub("", block).strip()
            body_words = len(re.findall(r"[A-Za-z0-9]+", body))
            has_numbers = bool(_NUMBER_PATTERN.search(body))
            is_not_found = (
                not has_numbers
                and body_words <= _MAX_NOT_FOUND_WORDS
                and bool(_NOT_FOUND_PATTERN.search(body))
            )

            if not citations:
                # Only claims, facts, and data need a citation. Navigation and
                # sentences that merely explain already-cited text do not.
                if (
                    is_not_found
                    or _is_signpost(body, has_numbers, body_words)
                    or _restates_cited_text(body, has_numbers, cited_words)
                ):
                    non_claim_blocks += 1
                    continue
                if body_words >= _MIN_UNCITED_WORDS or has_numbers:
                    factual_blocks += 1
                    uncited_blocks.append(block)
                continue

            factual_blocks += 1
            cited_blocks += 1
            cited_words |= _content_words(body)
            for source, page_text in citations:
                source = source.strip()
                page = int(page_text)
                if source not in pages_by_source:
                    issues.append(f"Unknown source cited: {source}, p. {page}")
                    continue
                if page not in pages_by_source[source]:
                    issues.append(f"Page not retrieved for source {source}: p. {page}")

            self._check_numbers(body, citations, chunks, issues)

        coverage = cited_blocks / factual_blocks if factual_blocks else 1.0
        logs.append(
            f"Citation coverage: {cited_blocks}/{factual_blocks} blocks ({coverage:.0%}), "
            f"required {_MIN_CITED_BLOCK_RATIO:.0%}"
        )
        if coverage < _MIN_CITED_BLOCK_RATIO:
            issues.extend(f"Uncited claim: \"{self._short(block)}\"" for block in uncited_blocks)
        elif uncited_blocks:
            logs.append(f"Uncited blocks allowed by coverage rule: {len(uncited_blocks)}")

        if non_claim_blocks:
            logs.append(f"Non-claim blocks skipped (navigation, restatements, lead-ins): {non_claim_blocks}")
        logs.append(f"Rule checks complete: {len(issues)} issue(s)")
        return issues

    def _check_numbers(
        self,
        body: str,
        citations: List[Tuple[str, str]],
        chunks: List[RetrievedChunk],
        issues: List[str],
    ) -> None:
        numbers = _extract_numbers(body)
        if not numbers:
            return

        cited_keys = {(src.strip(), int(pg)) for src, pg in citations}
        cited_text = " ".join(
            c.content for c in chunks if (c.source_file, c.page_number) in cited_keys
        )
        cited_numbers = _extract_numbers(cited_text)
        all_numbers = _extract_numbers(" ".join(c.content for c in chunks))

        for number in sorted(numbers):
            if number in cited_numbers:
                continue
            label = f"{number:g}"
            if number in all_numbers:
                issues.append(f"Citation mismatch: {label} is not on the cited page")
            else:
                issues.append(f"Unsupported number: {label} does not appear in any source")

    def _llm_check(
        self, draft: str, chunks: List[RetrievedChunk], logs: List[str]
    ) -> Tuple[bool, str]:
        user_prompt = (
            f"Sources:\n{format_sources(chunks)}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Return the JSON verdict now."
        )
        raw = self.llm.invoke(FACT_CHECK_SYSTEM_PROMPT, user_prompt)
        verdict = self._parse_verdict(raw)
        if verdict is None:
            logs.append("LLM support check returned an unparseable response; not approved")
            return False, "The support check could not be parsed. Make every claim explicitly supported."

        approved = bool(verdict.get("is_approved", False))
        critique = str(verdict.get("critique", "") or "").strip()
        unsupported = verdict.get("unsupported_items") or []
        if unsupported:
            critique = "\n".join(
                part for part in [critique, "Unsupported items: " + "; ".join(map(str, unsupported))] if part
            )
        logs.append(f"LLM support check: approved={approved}")
        return approved, critique

    @staticmethod
    def _parse_verdict(raw: str):
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _short(text: str, limit: int = 80) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 3] + "..."
