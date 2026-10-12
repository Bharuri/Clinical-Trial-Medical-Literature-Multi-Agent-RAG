"""
Retrieval and agent pipeline: retrieve, draft, fact-check, and revise.

Usage:
    python -m src.retrieval_agent.run "question" --retrieve-only
    python -m src.retrieval_agent.run "question" --retrieve-only --rerank   (re-ranks 200 candidates, keeps 10)
    python -m src.retrieval_agent.run "question"   (needs AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)

Full mode reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION and
BEDROCK_MODEL_ID from the environment. Region and model fall back to config.yaml.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
from src.retrieval_agent.bedrock_client import BedrockError, BedrockLLM, TextModel
from src.retrieval_agent.fact_check import FactCheckResult, FactChecker
from src.retrieval_agent.hybrid import HybridRetriever
from src.retrieval_agent.retrieval import RetrievalError, RetrievalResult, Retriever
from src.retrieval_agent.synthesis import SynthesisAgent


@dataclass
class DraftAttempt:
    """One drafted version and the fact-check verdict for that exact version."""

    label: str
    draft: str
    fact_check: FactCheckResult


@dataclass
class PipelineResult:
    question: str
    retrieval: Optional[RetrievalResult] = None
    draft: Optional[str] = None
    fact_check: Optional[FactCheckResult] = None
    revisions: int = 0
    final_answer: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    # Every version in order (initial draft first). draft/fact_check are the last entry.
    attempts: List[DraftAttempt] = field(default_factory=list)

    @property
    def is_approved(self) -> bool:
        return self.fact_check is not None and self.fact_check.is_approved


def run_pipeline(
    question: str,
    *,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    model_id: Optional[str] = None,
    llm: Optional[TextModel] = None,
    retriever: Optional[HybridRetriever] = None,
    top_k: Optional[int] = None,
    source_files: Optional[Sequence[str]] = None,
    use_reranker: bool = False,
    max_revisions: int = 2,
) -> PipelineResult:
    """
    Answer a question from the stored chunks with a fact-checked, cited draft.

    The Bedrock client is created before retrieval so missing credentials fail fast.
    A draft is returned as final only if the fact-checker approves it.
    source_files limits retrieval to those documents. None searches all of them.
    use_reranker re-ranks hybrid-search candidates with the cross-encoder.

    Raises:
        ValueError: If max_revisions is negative.
        BedrockError: If the Bedrock client cannot be created.
        RetrievalError: If retrieval fails.
    """
    if max_revisions < 0:
        raise ValueError(f"max_revisions must be >= 0, got {max_revisions}")

    logs: List[str] = []
    if llm is None:
        llm = BedrockLLM.from_credentials(
            aws_access_key_id, aws_secret_access_key, aws_region_name, model_id
        )
        logs.append("Bedrock client ready")

    owns_retriever = retriever is None
    if owns_retriever:
        retriever = HybridRetriever(Retriever(top_k=top_k))
    try:
        retrieval = retriever.retrieve(
            question, top_k=top_k, source_files=source_files, use_reranker=use_reranker
        )
    finally:
        if owns_retriever:
            retriever.close()
    reranker_note = "re-ranker on" if use_reranker else "re-ranker off"
    logs.append(
        f"Retrieved {retrieval.match_count} chunk(s) via route '{retrieval.route}' "
        f"(hybrid search, {reranker_note})"
    )

    result = PipelineResult(question=question, retrieval=retrieval, logs=logs)
    if not retrieval.chunks:
        logs.append("No chunks retrieved; skipping synthesis")
        return result

    synthesis = SynthesisAgent(llm)
    checker = FactChecker(llm)

    draft = synthesis.draft(question, retrieval.chunks)
    fact_check = checker.check(draft, retrieval.chunks)
    attempts = [DraftAttempt("Initial draft", draft, fact_check)]
    logs.append(f"Draft 0 checked: approved={fact_check.is_approved}")

    revisions = 0
    while not fact_check.is_approved and revisions < max_revisions:
        revisions += 1
        draft = synthesis.revise(question, retrieval.chunks, draft, fact_check.critique)
        fact_check = checker.check(draft, retrieval.chunks)
        attempts.append(DraftAttempt(f"Revision {revisions}", draft, fact_check))
        logs.append(f"Revision {revisions} checked: approved={fact_check.is_approved}")

    result.attempts = attempts
    result.draft = draft
    result.fact_check = fact_check
    result.revisions = revisions
    result.final_answer = draft if fact_check.is_approved else None
    logs.extend(fact_check.logs)
    return result


def _print_retrieval(retrieval: RetrievalResult) -> None:
    print(f"Route: {retrieval.route} | query dimensions: {retrieval.query_dimensions} | matches: {retrieval.match_count}")
    for index, chunk in enumerate(retrieval.chunks, start=1):
        print(
            f"\n[{index}] score={chunk.similarity_score:.4f} | {chunk.source_file} | "
            f"p. {chunk.page_number} | {chunk.chunk_type} | {chunk.section_header}"
        )
        print(f"    {chunk.preview()}")


def _print_result(result: PipelineResult) -> None:
    if result.retrieval is not None:
        _print_retrieval(result.retrieval)
    if result.final_answer is not None:
        print("\n=== Answer (fact-check approved) ===")
        print(result.final_answer)
    elif result.draft is not None:
        print("\n=== Draft NOT approved by fact-checker ===")
        print(result.draft)
        if result.fact_check and result.fact_check.critique:
            print("\n--- Critique ---")
            print(result.fact_check.critique)
    else:
        print("\nNo answer generated: no matching chunks were retrieved.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question against the clinical trial index.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve (ignored with --rerank)")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Re-rank 200 hybrid-search candidates with a cross-encoder and keep the best 10",
    )
    parser.add_argument("--max-revisions", type=int, default=2, help="Maximum revision rounds after a failed fact-check")
    parser.add_argument("--retrieve-only", action="store_true", help="Show retrieved chunks without calling Bedrock")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.retrieve_only:
            retriever = HybridRetriever(Retriever(top_k=args.top_k))
            try:
                _print_retrieval(
                    retriever.retrieve(args.question, top_k=args.top_k, use_reranker=args.rerank)
                )
            finally:
                retriever.close()
            return 0

        result = run_pipeline(
            args.question,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_region_name=os.environ.get("AWS_REGION"),
            model_id=os.environ.get("BEDROCK_MODEL_ID"),
            top_k=args.top_k,
            use_reranker=args.rerank,
            max_revisions=args.max_revisions,
        )
    except (BedrockError, RetrievalError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    _print_result(result)
    return 0 if result.is_approved else 1


if __name__ == "__main__":
    sys.exit(main())
