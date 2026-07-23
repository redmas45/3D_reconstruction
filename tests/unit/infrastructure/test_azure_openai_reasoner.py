import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infrastructure.azure_openai_reasoner import (
    AzureReasoningSettings,
    probe_azure_reasoning,
    request_decision_trace,
    request_gap_decisions,
)


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

    def test_gap_request_includes_bounded_image_as_low_detail_data_url(self) -> None:
        settings = AzureReasoningSettings(
            endpoint="https://example.openai.azure.com",
            api_key="secret-test-key",
            deployment="gpt-5.4",
        )
        response_payload = {
            "schema_version": 2,
            "evidence_digest": "evidence",
            "clue_digest": "clues",
            "hypothesis_digest": "hypotheses",
            "decisions": [],
        }
        response = _Response({"id": "response-2", "output_text": json.dumps(response_payload)})
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "visible.jpg"
            image_path.write_bytes(b"visible-image")
            with patch("urllib.request.urlopen", return_value=response) as urlopen_mock:
                returned, _ = request_gap_decisions(
                    settings,
                    {"evidence_policy": "visible_only"},
                    {"type": "object"},
                    [str(image_path)],
                    "low",
                )
        request_body = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
        image_content = request_body["input"][0]["content"][1]
        self.assertEqual("input_image", image_content["type"])
        self.assertEqual("low", image_content["detail"])
        self.assertTrue(image_content["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(response_payload, returned)

    def test_probe_uses_small_non_chain_of_thought_request(self) -> None:
        settings = AzureReasoningSettings(
            endpoint="https://example.openai.azure.com",
            api_key="secret-test-key",
            deployment="gpt-5.4",
            max_output_tokens=8_000,
            reasoning_effort="medium",
        )
        response = _Response({
            "id": "response-probe",
            "output_text": json.dumps({"status": "ready"}),
        })

        with patch("urllib.request.urlopen", return_value=response) as urlopen_mock:
            metadata = probe_azure_reasoning(settings)

        request_body = json.loads(urlopen_mock.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(512, request_body["max_output_tokens"])
        self.assertEqual("none", request_body["reasoning"]["effort"])
        self.assertEqual("response-probe", metadata["response_id"])


if __name__ == "__main__":
    unittest.main()
