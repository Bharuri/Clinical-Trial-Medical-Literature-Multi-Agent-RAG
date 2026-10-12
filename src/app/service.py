"""
Bridge between the Streamlit UI and the retrieval/agent pipeline.

All retrieval and agent logic stays in src/retrieval_agent. This module only
validates the request, shares one embedding model per server process, and
closes the DuckDB connection it opened.
"""

from typing import Optional, Sequence

import streamlit as st

from src.app.accepted_answers import load_accepted_answers
from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval_agent.answer_cache import CachedAnswer, find_cached_answer
from src.retrieval_agent.bedrock_client import BedrockLLM, TextModel
from src.retrieval_agent.hybrid import HybridRetriever
from src.retrieval_agent.reranker import CrossEncoderReranker, Reranker
from src.retrieval_agent.retrieval import Retriever
from src.retrieval_agent.run import PipelineResult, run_pipeline
from src.utils.config import Config


@st.cache_resource(show_spinner=False)
def get_query_encoder() -> EmbeddingGenerator:
    """Load the Sentence Transformers model once per server process, not per user session."""
    config = Config.get_config()
    return EmbeddingGenerator(
        model_name=config.get_embedding_model(),
        expected_dimensions=config.get_embedding_dimensions(),
    )


@st.cache_resource(show_spinner=False)
def get_reranker() -> Reranker:
    """Load the cross-encoder re-ranker once per server process, and only when it is first requested."""
    return CrossEncoderReranker(Config.get_config().get_reranker_model())


def find_accepted_answer(question: str, db_path: Optional[str] = None) -> Optional[CachedAnswer]:
    """
    Look up a saved answer for the question before the LLM pipeline runs.

    The embedding model loads only when no exact match exists and at least one
    answer has been saved.

    Returns:
        A CachedAnswer for an exact or similar saved question, or None.
    """
    if not question.strip():
        return None
    saved = load_accepted_answers(db_path=db_path)
    if not saved:
        return None
    return find_cached_answer(
        question,
        saved,
        lambda texts: get_query_encoder().encode_texts(texts),
    )


def answer_question(
    question: str,
    *,
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_region_name: Optional[str],
    model_id: Optional[str],
    source_files: Optional[Sequence[str]] = None,
    use_reranker: bool = False,
    llm: Optional[TextModel] = None,
    retriever: Optional[HybridRetriever] = None,
) -> PipelineResult:
    """
    Run the multi-agent pipeline for one question.

    Input and credential checks run before the embedding model is loaded.
    source_files limits retrieval to those documents. None searches all of them.
    use_reranker re-ranks hybrid-search candidates with the cross-encoder.
    The llm and retriever arguments exist so tests can inject fakes.

    Raises:
        ValueError: If the question is empty, or source_files is an empty list.
        BedrockError: If the Bedrock client cannot be created.
        RetrievalError, EmbeddingError: If retrieval cannot run.
    """
    if not question or not question.strip():
        raise ValueError("Enter a research question before running the pipeline.")
    if source_files is not None and len(source_files) == 0:
        raise ValueError("Select at least one document to search.")

    if llm is None:
        llm = BedrockLLM.from_credentials(
            aws_access_key_id, aws_secret_access_key, aws_region_name, model_id
        )

    owns_retriever = retriever is None
    if owns_retriever:
        retriever = HybridRetriever(
            Retriever(encoder=get_query_encoder()),
            reranker_factory=get_reranker,
        )
    try:
        return run_pipeline(
            question,
            llm=llm,
            retriever=retriever,
            source_files=source_files,
            use_reranker=use_reranker,
        )
    finally:
        if owns_retriever:
            retriever.close()
