"""
Bridge between the Streamlit UI and the retrieval/agent pipeline.

All retrieval and agent logic stays in src/retrieval_agent. This module only
validates the request, shares one embedding model per server process, and
closes the DuckDB connection it opened.
"""

from typing import Optional

import streamlit as st

from src.ingestion.embeddings import EmbeddingGenerator
from src.retrieval_agent.bedrock_client import BedrockLLM, TextModel
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


def answer_question(
    question: str,
    *,
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_region_name: Optional[str],
    model_id: Optional[str],
    llm: Optional[TextModel] = None,
    retriever: Optional[Retriever] = None,
) -> PipelineResult:
    """
    Run the multi-agent pipeline for one question.

    Input and credential checks run before the embedding model is loaded.
    The llm and retriever arguments exist so tests can inject fakes.

    Raises:
        ValueError: If the question is empty.
        BedrockError: If the Bedrock client cannot be created.
        RetrievalError, EmbeddingError: If retrieval cannot run.
    """
    if not question or not question.strip():
        raise ValueError("Enter a research question before running the pipeline.")

    if llm is None:
        llm = BedrockLLM.from_credentials(
            aws_access_key_id, aws_secret_access_key, aws_region_name, model_id
        )

    owns_retriever = retriever is None
    if owns_retriever:
        retriever = Retriever(encoder=get_query_encoder())
    try:
        return run_pipeline(question, llm=llm, retriever=retriever)
    finally:
        if owns_retriever:
            retriever.close()
