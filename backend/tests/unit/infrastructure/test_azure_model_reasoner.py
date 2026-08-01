import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from infrastructure.azure_model_reasoner import (
    AzureReasoningConfigurationError,
    AzureReasoningSettings,
    probe_azure_reasoning,
    request_decision_trace,
    request_gap_decisions,
    request_reconstruction_narrative,
    request_structured_response,
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


class AzureModelReasonerTests(unittest.TestCase):
    def test_reads_current_grok_environment_contract(self) -> None:
        environment = {
            "AZURE_GROK_BASE_URL": "https://example.services.ai.azure.com/openai/v1",
            "AZURE_GROK_API_KEY": "secret-test-key",
            "AZURE_GROK_CHAT_DEPLOYMENT": "grok-4-20-reasoning",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = AzureReasoningSettings.from_environment({})

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual("secret-test-key", settings.api_key)
        self.assertEqual("grok-4-20-reasoning", settings.deployment)

    def test_foundry_grok_uses_responses_endpoint(self) -> None:
        settings = AzureReasoningSettings(
            endpoint="https://example.services.ai.azure.com/openai/v1",
            api_key="secret-test-key",
            deployment="grok-4-20-reasoning",
        )
        response = _Response({"id": "response-1", "output_text": '{"status":"ready"}'})
        with patch("urllib.request.urlopen", return_value=response) as urlopen_mock:
            returned, metadata = request_structured_response(
                settings,
                "Return readiness.",
                {"operation": "probe"},
                "probe",
                {"type": "object"},
                "Return JSON only.",
            )

        request = urlopen_mock.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual("https://example.services.ai.azure.com/openai/v1/responses", request.full_url)
        self.assertEqual("grok-4-20-reasoning", request_body["model"])
        self.assertEqual("secret-test-key", request.headers["Api-key"])
        self.assertEqual({"status": "ready"}, returned)
        self.assertEqual("azure_foundry_openai_v1", metadata["provider"])

    def test_uses_chat_deployment_and_responses_endpoint_without_exposing_key(self) -> None:
        environment = {
            "AZURE_GROK_BASE_URL": "https://example.services.ai.azure.com/openai/v1",
            "AZURE_GROK_API_KEY": "secret-test-key",
            "AZURE_GROK_CHAT_DEPLOYMENT": "grok-4-20-reasoning",
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
        self.assertEqual("https://example.services.ai.azure.com/openai/v1/responses", request.full_url)
        self.assertEqual("grok-4-20-reasoning", request_body["model"])
        self.assertEqual("secret-test-key", request.headers["Api-key"])
        self.assertNotIn("secret-test-key", json.dumps(metadata))
        self.assertEqual(trace, returned_trace)

    def test_gap_request_includes_bounded_image_as_low_detail_data_url(self) -> None:
        settings = AzureReasoningSettings(
            endpoint="https://example.services.ai.azure.com/openai/v1",
            api_key="secret-test-key",
            deployment="grok-4-20-reasoning",
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
            endpoint="https://example.services.ai.azure.com/openai/v1",
            api_key="secret-test-key",
            deployment="grok-4-20-reasoning",
            max_output_tokens=8_000,
            reasoning_effort="medium",
        )
        response = _Response({
            "id": "response-probe",
            "output_text": json.dumps({"status": "ready"}),
        })

        with patch("urllib.request.urlopen", return_value=response) as urlopen_mock:
            metadata = probe_azure_reasoning(settings)

        request_body = json.loads(
            urlopen_mock.call_args.args[0].data.decode("utf-8")
        )
        self.assertEqual(512, request_body["max_output_tokens"])
        self.assertNotIn("reasoning", request_body)
        self.assertEqual("response-probe", metadata["response_id"])
        self.assertTrue(metadata["deployment_validated"])

    def test_probe_reports_available_deployments_before_model_request(self) -> None:
        settings = AzureReasoningSettings(
            endpoint="https://example.openai.azure.com/openai/v1/",
            api_key="secret-test-key",
            deployment="missing-deployment",
        )
        deployment_response = _Response({
            "data": [{"id": "gpt-5-mini"}, {"id": "gpt-5.4-mini"}],
        })

        with patch("urllib.request.urlopen", return_value=deployment_response):
            with self.assertRaisesRegex(
                AzureReasoningConfigurationError,
                "Available deployments: gpt-5-mini, gpt-5.4-mini",
            ):
                probe_azure_reasoning(settings)

    def test_narrative_normalizes_foundry_presentation_text_to_project_contract(self) -> None:
        settings = AzureReasoningSettings(
            endpoint="https://example.services.ai.azure.com/openai/v1",
            api_key="secret-test-key",
            deployment="grok-4-20-reasoning",
        )
        response = _Response({
            "id": "response-narrative",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "**Reconstruction Summary**\n\nA person continues left to right.",
                }],
            }],
        })
        payload = {
            "clue_digest": "digest",
            "clues": [{"id": "clue_01", "statement": "Visible motion", "confidence": 0.9}],
            "gap_decisions": [{
                "gap_index": 0,
                "gap_summary": "The person continues left to right.",
                "clue_ids": ["clue_01"],
                "confidence": 0.8,
                "unknowns": ["Exact speed is uncertain."],
            }],
        }
        schema = {
            "type": "object",
            "properties": {
                "schema_version": {"const": 1},
                "mode": {"enum": ["azure"]},
            },
        }
        with patch("urllib.request.urlopen", return_value=response):
            narrative, metadata = request_reconstruction_narrative(settings, payload, schema)

        self.assertEqual(1, narrative["schema_version"])
        self.assertEqual("azure", narrative["mode"])
        self.assertEqual(1, len(narrative["gap_summaries"]))
        self.assertTrue(narrative["whole_video_summary"])
        self.assertEqual("response-narrative", metadata["response_id"])


if __name__ == "__main__":
    unittest.main()
