"""
Tests for the Bedrock wrapper used by the synthesis and fact-check agents.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval_agent.bedrock_client import BedrockError, BedrockLLM
from src.utils.config import Config


@pytest.fixture(autouse=True)
def reset_config():
    Config.reset()
    yield
    Config.reset()


def _converse_response(text: str) -> dict:
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}}


class TestInvoke:
    def test_sends_system_and_user_prompts(self):
        client = MagicMock()
        client.converse.return_value = _converse_response("answer")
        llm = BedrockLLM(client, "model-x", max_tokens=100, temperature=0.0)

        llm.invoke("system text", "user text")

        client.converse.assert_called_once_with(
            modelId="model-x",
            system=[{"text": "system text"}],
            messages=[{"role": "user", "content": [{"text": "user text"}]}],
            inferenceConfig={"maxTokens": 100, "temperature": 0.0},
        )

    def test_returns_stripped_text(self):
        client = MagicMock()
        client.converse.return_value = _converse_response("  the answer \n")
        llm = BedrockLLM(client, "model-x")

        assert llm.invoke("s", "u") == "the answer"

    def test_joins_multiple_text_blocks(self):
        client = MagicMock()
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "part one. "}, {"text": "part two."}]}}
        }
        llm = BedrockLLM(client, "model-x")

        assert llm.invoke("s", "u") == "part one. part two."

    def test_empty_text_raises(self):
        client = MagicMock()
        client.converse.return_value = _converse_response("   ")
        llm = BedrockLLM(client, "model-x")

        with pytest.raises(BedrockError, match="empty"):
            llm.invoke("s", "u")

    def test_bad_response_shape_raises(self):
        client = MagicMock()
        client.converse.return_value = {"unexpected": True}
        llm = BedrockLLM(client, "model-x")

        with pytest.raises(BedrockError, match="shape"):
            llm.invoke("s", "u")

    def test_client_exception_is_wrapped(self):
        client = MagicMock()
        client.converse.side_effect = RuntimeError("throttled")
        llm = BedrockLLM(client, "model-x")

        with pytest.raises(BedrockError, match="throttled"):
            llm.invoke("s", "u")

    def test_model_id_required(self):
        with pytest.raises(BedrockError, match="model ID"):
            BedrockLLM(MagicMock(), "")


class TestFromCredentials:
    def test_missing_credentials_raise_before_client_use(self):
        with pytest.raises(BedrockError):
            BedrockLLM.from_credentials("", "", "us-east-1", "model-x")

    def test_none_credentials_raise(self):
        with pytest.raises(BedrockError):
            BedrockLLM.from_credentials(None, None, None, None)

    def test_uses_config_defaults_for_region_and_model(self):
        fake_client = MagicMock()
        with patch(
            "src.retrieval_agent.bedrock_client.BedrockClient.create_client",
            return_value=fake_client,
        ) as create:
            llm = BedrockLLM.from_credentials("key", "secret")

        config = Config.get_config()
        create.assert_called_once_with("key", "secret", config.get_default_region())
        assert llm.model_id == config.get_default_bedrock_model()
        assert llm.client is fake_client

    def test_explicit_region_and_model_override_defaults(self):
        with patch(
            "src.retrieval_agent.bedrock_client.BedrockClient.create_client",
            return_value=MagicMock(),
        ) as create:
            llm = BedrockLLM.from_credentials("key", "secret", "eu-west-1", "custom-model")

        create.assert_called_once_with("key", "secret", "eu-west-1")
        assert llm.model_id == "custom-model"


class TestRealSdkClient:
    """Uses the real boto3 client with dummy credentials. No network call is made."""

    def test_real_client_exposes_converse(self):
        llm = BedrockLLM.from_credentials("AKIATEST", "secret", "us-east-1", "test-model")
        assert callable(getattr(llm.client, "converse", None))
