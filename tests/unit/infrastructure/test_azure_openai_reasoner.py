import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.azure_openai_reasoner import AzureReasoningSettings, request_decision_trace


class _Response:
    def __init__(self, payload: dict) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.read()


class AzureOpenAIReasonerTests(unittest.TestCase):
    def test_uses_chat_deployment_and_responses_endpoint_without_exposing_key(self) -> None:
        environment = {
            "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "secret-test-key",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "gpt-5.4",
        }
        trace = {"schema_version": 1, "evidence_digest": "abc", "decisions": []}
        response = _Response({"id": "response-1", "output_text": json.dumps(trace), "usage": {"total_tokens": 12}})
        with patch.dict(os.environ, environment, clear=True):
            settings = AzureReasoningSettings.from_environment({})
        self.assertIsNotNone(settings)
        assert settings is not None

        with patch("urllib.request.urlopen", return_value=response) as urlopen_mock:
            returned_trace, metadata = request_decision_trace(settings, {"evidence_digest": "abc"}, {"type": "object"})

        request = urlopen_mock.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual("https://example.openai.azure.com/openai/v1/responses", request.full_url)
        self.assertEqual("gpt-5.4", request_body["model"])
        self.assertEqual("secret-test-key", request.headers["Api-key"])
        self.assertNotIn("secret-test-key", json.dumps(metadata))
        self.assertEqual(trace, returned_trace)


if __name__ == "__main__":
    unittest.main()
