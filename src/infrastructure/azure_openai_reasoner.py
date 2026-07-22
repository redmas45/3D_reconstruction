"""Calls Azure OpenAI Responses API with a strictly structured output contract."""

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse


AZURE_BASE_URL_VARIABLE = "AZURE_OPENAI_BASE_URL"
AZURE_ENDPOINT_VARIABLE = "AZURE_OPENAI_ENDPOINT"
AZURE_KEY_VARIABLE = "AZURE_OPENAI_API_KEY"
AZURE_CHAT_DEPLOYMENT_VARIABLE = "AZURE_OPENAI_CHAT_DEPLOYMENT"
LEGACY_DEPLOYMENT_VARIABLE = "AZURE_OPENAI_DEPLOYMENT"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_TOKENS = 8_000
SUPPORTED_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
SYSTEM_INSTRUCTIONS = """You are a constrained forensic reconstruction planner.
Use only the supplied visible-frame evidence ledger. Select exactly one supplied hypothesis per gap.
Never invent evidence, coordinates, identities, events, or hidden-frame observations.
Prefer the measured continuation when evidence supports it; choose conservative motion when boundary
evidence conflicts. Report uncertainty plainly. Return only the required structured decision trace.
Do not provide hidden chain-of-thought; decision_summary and rejection reasons must be concise evidence-grounded conclusions."""


class AzureReasoningConfigurationError(ValueError):
    pass


class AzureReasoningRequestError(RuntimeError):
    pass


class AzureReasoningResponseError(RuntimeError):
    pass


@dataclass(frozen=True)
class AzureReasoningSettings:
    endpoint: str
    api_key: str
    deployment: str
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    reasoning_effort: str = "medium"

    @classmethod
    def from_environment(cls, configuration: dict) -> "AzureReasoningSettings | None":
        endpoint = (
            os.environ.get(AZURE_BASE_URL_VARIABLE, "").strip()
            or os.environ.get(AZURE_ENDPOINT_VARIABLE, "").strip()
        )
        api_key = os.environ.get(AZURE_KEY_VARIABLE, "").strip()
        deployment = (
            os.environ.get(AZURE_CHAT_DEPLOYMENT_VARIABLE, "").strip()
            or os.environ.get(LEGACY_DEPLOYMENT_VARIABLE, "").strip()
        )
        configured_values = (endpoint, api_key, deployment)
        if not any(configured_values):
            return None
        if not all(configured_values):
            raise AzureReasoningConfigurationError(
                "Azure reasoning requires endpoint, API key, and chat deployment environment values"
            )
        settings = cls(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            timeout_seconds=int(configuration.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)),
            max_output_tokens=int(configuration.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
            reasoning_effort=str(configuration.get("reasoning_effort", "medium")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed_endpoint = urlparse(self.endpoint)
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
            raise AzureReasoningConfigurationError("AZURE_OPENAI_BASE_URL must be an HTTPS URL")
        if not self.deployment or len(self.deployment) > 128:
            raise AzureReasoningConfigurationError("Azure chat deployment name is invalid")
        if not 10 <= self.timeout_seconds <= 600:
            raise AzureReasoningConfigurationError("Azure reasoning timeout must be between 10 and 600 seconds")
        if not 512 <= self.max_output_tokens <= 32_000:
            raise AzureReasoningConfigurationError("Azure max output tokens must be between 512 and 32000")
        if self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise AzureReasoningConfigurationError("Azure reasoning effort is unsupported")

    @property
    def responses_url(self) -> str:
        base_url = self.endpoint.rstrip("/")
        if base_url.endswith("/openai/v1"):
            return f"{base_url}/responses"
        return f"{base_url}/openai/v1/responses"


def request_decision_trace(settings: AzureReasoningSettings, ledger: dict, schema: dict) -> tuple[dict, dict]:
    request_payload = _request_payload(settings, ledger, schema)
    request = urllib.request.Request(
        settings.responses_url,
        data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": settings.api_key},
        method="POST",
    )
    response_payload = _send_request(request, settings.timeout_seconds)
    trace = _parse_structured_output(response_payload)
    return trace, _response_metadata(response_payload, settings.deployment)


def _request_payload(settings: AzureReasoningSettings, ledger: dict, schema: dict) -> dict:
    return {
        "model": settings.deployment,
        "store": False,
        "instructions": SYSTEM_INSTRUCTIONS,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Select bounded reconstruction hypotheses for this evidence ledger:\n" + json.dumps(ledger),
            }],
        }],
        "reasoning": {"effort": settings.reasoning_effort},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "reconstruction_decision_trace",
                "strict": True,
                "schema": schema,
            },
        },
        "max_output_tokens": settings.max_output_tokens,
    }


def _send_request(request: urllib.request.Request, timeout_seconds: int) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        raise AzureReasoningRequestError(f"Azure reasoning request failed with HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        raise AzureReasoningRequestError("Azure reasoning request could not be completed") from error
    try:
        payload = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AzureReasoningResponseError("Azure reasoning returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise AzureReasoningResponseError("Azure reasoning response must be an object")
    return payload


def _parse_structured_output(response_payload: dict) -> dict:
    output_text = response_payload.get("output_text")
    if not isinstance(output_text, str):
        output_text = _find_output_text(response_payload.get("output"))
    if not output_text:
        raise AzureReasoningResponseError("Azure reasoning response contained no structured output")
    try:
        trace = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise AzureReasoningResponseError("Azure reasoning structured output was invalid JSON") from error
    if not isinstance(trace, dict):
        raise AzureReasoningResponseError("Azure reasoning structured output must be an object")
    return trace


def _find_output_text(output: object) -> str | None:
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, dict) or not isinstance(item.get("content"), list):
            continue
        for content in item["content"]:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    return None


def _response_metadata(response_payload: dict, deployment: str) -> dict:
    usage = response_payload.get("usage")
    sanitized_usage = usage if isinstance(usage, dict) else {}
    return {
        "provider": "azure_openai",
        "deployment": deployment,
        "response_id": str(response_payload.get("id", "")),
        "usage": sanitized_usage,
    }
