"""
Synthesis agent: drafts a cited answer from retrieved chunks and revises it
when the fact-checker returns a critique.
"""

import re
from typing import Dict, List, Set

from src.retrieval_agent.bedrock_client import TextModel
from src.retrieval_agent.retrieval import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+?),\s*p\.\s*(\d+)\s*\]")

SYNTHESIS_SYSTEM_PROMPT = """You are the synthesis agent for a clinical trial and medical literature assistant.
Answer the user's question using ONLY the numbered sources provided.

Rules:
- Write an objective, professional, scientifically rigorous answer in Markdown.
- Use ONLY facts stated in the sources. Do not add background knowledge, framing, or conclusions the sources do not state.
- Every bullet or paragraph that states facts must include a citation in exactly this format: [Source: <source_file>, p. <page_number>]. One citation covers the sentences in that bullet or paragraph, so put it at the end of the block. Every sentence in a block must come from the page cited for that block.
- In each citation, <source_file> is the source_file value from the source header (for example, Clinical-Trials.pdf). Never use a list number or other index as the source.
- Copy numerical values exactly as they appear in the sources.
- For Markdown tables, keep each value in its original row and column.
- If the sources do not contain the answer, reply with exactly: The provided sources do not contain information to answer this question.
- Output only the answer."""


def format_sources(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks with the metadata the citations must use."""
    blocks = []
    for chunk in chunks:
        header = (
            f"source_file: {chunk.source_file} | page_number: {chunk.page_number} "
            f"| chunk_type: {chunk.chunk_type} | section_header: {chunk.section_header}"
        )
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n".join(blocks)


def _source_key(name: str) -> str:
    """Reduce a filename to letters and digits so case and separator differences compare equal."""
    base = re.sub(r"\.pdf$", "", name.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", base.lower())


def canonicalize_citations(draft: str, chunks: List[RetrievedChunk]) -> str:
    """
    Rewrite citation filenames to the stored source_file when they differ only in case or separators.

    Names that match no stored source, or match more than one, are left unchanged so the
    fact-checker still flags them.
    """
    stored = {chunk.source_file for chunk in chunks}
    names_by_key: Dict[str, Set[str]] = {}
    for name in stored:
        names_by_key.setdefault(_source_key(name), set()).add(name)

    def replace(match: "re.Match[str]") -> str:
        source = match.group(1).strip()
        if source in stored:
            return match.group(0)
        candidates = names_by_key.get(_source_key(source), set())
        if len(candidates) != 1:
            return match.group(0)
        (exact,) = candidates
        return f"[Source: {exact}, p. {match.group(2)}]"

    return CITATION_PATTERN.sub(replace, draft)


class SynthesisAgent:
    """Drafts and revises cited answers with a Bedrock text model."""

    def __init__(self, llm: TextModel):
        self.llm = llm

    def draft(self, question: str, chunks: List[RetrievedChunk]) -> str:
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Sources:\n{format_sources(chunks)}\n\n"
            "Write the cited answer now."
        )
        return canonicalize_citations(self._generate(user_prompt), chunks)

    def revise(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        previous_draft: str,
        critique: str,
    ) -> str:
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Sources:\n{format_sources(chunks)}\n\n"
            f"Previous draft:\n{previous_draft}\n\n"
            f"Fact-checker critique (fix every point):\n{critique}\n\n"
            "For each uncited or unsupported sentence in the critique, either add a citation "
            "from the sources that supports it, or delete the sentence. Do not keep a flagged "
            "sentence unchanged. Write the corrected cited answer now."
        )
        return canonicalize_citations(self._generate(user_prompt), chunks)

    def _generate(self, user_prompt: str) -> str:
        return self.llm.invoke(SYNTHESIS_SYSTEM_PROMPT, user_prompt).strip()
