"""
Amazon Bedrock runtime wrapper for the synthesis and fact-checking agents.

Credentials and region come from the request; the client is built with the
shared BedrockClient factory in src/utils/connections.py.
"""

from typing import Any, Optional, Protocol

from src.utils.config import Config
from src.utils.connections import BedrockClient


class BedrockError(Exception):
    """Raised when a Bedrock client cannot be created or a model call fails."""
    pass


class TextModel(Protocol):
    """Anything that turns a system prompt and user prompt into text."""

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        ...


class BedrockLLM:
    """Calls a Bedrock model through the Converse API."""

    def __init__(
        self,
        client: Any,
        model_id: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        if not model_id:
            raise BedrockError("Bedrock model ID is required")
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_credentials(
        cls,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_region_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> "BedrockLLM":
        """
        Build a model wrapper from per-request AWS settings.

        Region and model fall back to config.yaml defaults when omitted.
        """
        config = Config.get_config()
        region = aws_region_name or config.get_default_region()
        model = model_id or config.get_default_bedrock_model()
        try:
            client = BedrockClient.create_client(
                aws_access_key_id or "",
                aws_secret_access_key or "",
                region,
            )
        except ValueError as e:
            raise BedrockError(str(e)) from e
        return cls(client, model)

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send one system + user exchange to the model and return its text.

        Raises:
            BedrockError: If the call fails or the response has no text.
        """
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
        except Exception as e:
            raise BedrockError(f"Bedrock invocation failed: {e}") from e
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            blocks = response["output"]["message"]["content"]
        except (KeyError, TypeError) as e:
            raise BedrockError("Unexpected Bedrock response shape") from e

        text = "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict)
        ).strip()
        if not text:
            raise BedrockError("Bedrock returned an empty response")
        return text
