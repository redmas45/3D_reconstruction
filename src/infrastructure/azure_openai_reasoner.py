"""Calls Azure OpenAI Responses API with strictly structured output contracts."""

import base64
import json
import mimetypes
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from urllib.parse import urlparse


AZURE_BASE_URL_VARIABLE = "AZURE_OPENAI_BASE_URL"
AZURE_ENDPOINT_VARIABLE = "AZURE_OPENAI_ENDPOINT"
AZURE_KEY_VARIABLE = "AZURE_OPENAI_API_KEY"
AZURE_CHAT_DEPLOYMENT_VARIABLE = "AZURE_OPENAI_CHAT_DEPLOYMENT"
LEGACY_DEPLOYMENT_VARIABLE = "AZURE_OPENAI_DEPLOYMENT"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_TOKENS = 8_000
AZURE_DEPLOYMENTS_API_VERSION = "2023-03-15-preview"
MAXIMUM_AZURE_ERROR_MESSAGE_LENGTH = 400
SUPPORTED_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
SYSTEM_INSTRUCTIONS = """You are a constrained forensic reconstruction planner.
Use only the supplied visible-frame evidence ledger, clue catalog, bounded images, and hypothesis library.
Select exactly one supplied hypothesis for every supplied entity in every supplied gap.
Use each hypothesis's full supplied id in selected and rejected fields; never substitute its type name.
Never invent evidence, coordinates, identities, events, or hidden-frame observations.
Prefer the measured continuation when evidence supports it; choose conservative motion when boundary
evidence conflicts. Image inputs may support semantic interpretation but never override supplied numeric
measurements, counts, identity IDs, or paths. Report uncertainty plainly. Return only the required schema.
Do not provide hidden chain-of-thought; decision_summary and rejection reasons must be concise evidence-grounded conclusions."""
NARRATIVE_INSTRUCTIONS = """You create a concise presentation summary for an evidence-grounded reconstruction.
Use only supplied clues and validated gap decisions. Clearly distinguish observed boundary facts from inferred
inside-gap events. Do not claim recovered ground truth or invent causal links. Return only the required schema."""
SUPPORTED_IMAGE_DETAILS = frozenset({"low", "high", "original", "auto"})
PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ready"]},
    },
    "required": ["status"],
    "additionalProperties": False,
}


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

    @property
    def deployments_url(self) -> str:
        parsed_endpoint = urlparse(self.endpoint)
        authority = f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}"
        return (
            f"{authority}/openai/deployments"
            f"?api-version={AZURE_DEPLOYMENTS_API_VERSION}"
        )


def request_decision_trace(settings: AzureReasoningSettings, ledger: dict, schema: dict) -> tuple[dict, dict]:
    return request_structured_response(
        settings,
        prompt="Select bounded reconstruction hypotheses for this evidence ledger:",
        payload=ledger,
        schema_name="reconstruction_decision_trace",
        schema=schema,
        instructions=SYSTEM_INSTRUCTIONS,
    )


def request_gap_decisions(
    settings: AzureReasoningSettings,
    payload: dict,
    schema: dict,
    image_paths: list[str],
    image_detail: str,
) -> tuple[dict, dict]:
    return request_structured_response(
        settings,
        prompt="Select exactly one supplied hypothesis for every entity in every gap.",
        payload=payload,
        schema_name="gap_decisions_v2",
        schema=schema,
        instructions=SYSTEM_INSTRUCTIONS,
        image_paths=image_paths,
        image_detail=image_detail,
    )


def request_reconstruction_narrative(
    settings: AzureReasoningSettings,
    payload: dict,
    schema: dict,
) -> tuple[dict, dict]:
    return request_structured_response(
        settings,
        prompt="Summarize the validated reconstruction for presentation.",
        payload=payload,
        schema_name="reconstruction_narrative",
        schema=schema,
        instructions=NARRATIVE_INSTRUCTIONS,
    )


def probe_azure_reasoning(settings: AzureReasoningSettings) -> dict:
    deployment_metadata = probe_azure_deployment(settings)
    probe_settings = replace(
        settings,
        max_output_tokens=512,
        reasoning_effort="none",
    )
    response, metadata = request_structured_response(
        probe_settings,
        prompt="Return the required readiness status.",
        payload={"operation": "configuration_probe"},
        schema_name="azure_reasoning_probe",
        schema=PROBE_SCHEMA,
        instructions="Validate structured-output availability. Return only the required schema.",
    )
    if response.get("status") != "ready":
        raise AzureReasoningResponseError("Azure reasoning readiness probe was invalid")
    return {**metadata, **deployment_metadata}


def probe_azure_deployment(settings: AzureReasoningSettings) -> dict:
    request = urllib.request.Request(
        settings.deployments_url,
        headers={"api-key": settings.api_key},
        method="GET",
    )
    payload = _send_request(request, settings.timeout_seconds)
    deployment_names = _deployment_names(payload)
    if settings.deployment not in deployment_names:
        available = ", ".join(deployment_names) or "none"
        raise AzureReasoningConfigurationError(
            f"Azure deployment '{settings.deployment}' was not found. "
            f"Available deployments: {available}"
        )
    return {
        "deployment_validated": True,
        "available_deployment_count": len(deployment_names),
    }


def request_structured_response(
    settings: AzureReasoningSettings,
    prompt: str,
    payload: dict,
    schema_name: str,
    schema: dict,
    instructions: str,
    image_paths: list[str] | None = None,
    image_detail: str = "low",
) -> tuple[dict, dict]:
    request_payload = _request_payload(
        settings, prompt, payload, schema_name, schema, instructions,
        image_paths or [], image_detail,
    )
    request = urllib.request.Request(
        settings.responses_url,
        data=json.dumps(request_payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": settings.api_key},
        method="POST",
    )
    response_payload = _send_request(request, settings.timeout_seconds)
    structured_output = _parse_structured_output(response_payload)
    return structured_output, _response_metadata(response_payload, settings.deployment)


def _request_payload(
    settings: AzureReasoningSettings,
    prompt: str,
    payload: dict,
    schema_name: str,
    schema: dict,
    instructions: str,
    image_paths: list[str],
    image_detail: str,
) -> dict:
    content = [{"type": "input_text", "text": prompt + "\n" + json.dumps(payload)}]
    content.extend(_image_content(image_paths, image_detail))
    return {
        "model": settings.deployment,
        "store": False,
        "instructions": instructions,
        "input": [{
            "role": "user",
            "content": content,
        }],
        "reasoning": {"effort": settings.reasoning_effort},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        "max_output_tokens": settings.max_output_tokens,
    }


def _image_content(image_paths: list[str], image_detail: str) -> list[dict]:
    if image_detail not in SUPPORTED_IMAGE_DETAILS:
        raise AzureReasoningConfigurationError("Azure image detail is unsupported")
    return [
        {
            "type": "input_image",
            "image_url": _image_data_url(image_path),
            "detail": image_detail,
        }
        for image_path in image_paths
    ]


def _image_data_url(image_path: str) -> str:
    path = os.path.abspath(image_path)
    mime_type = mimetypes.guess_type(path)[0]
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise AzureReasoningConfigurationError("Azure visual evidence image type is unsupported")
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
    except OSError as error:
        raise AzureReasoningConfigurationError("Azure visual evidence image cannot be read") from error
    return f"data:{mime_type};base64,{encoded}"


def _send_request(request: urllib.request.Request, timeout_seconds: int) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        detail = _azure_http_error_detail(error)
        raise AzureReasoningRequestError(
            f"Azure reasoning request failed with HTTP {error.code}{detail}"
        ) from error
    except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
        raise AzureReasoningRequestError("Azure reasoning request could not be completed") from error
    try:
        payload = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AzureReasoningResponseError("Azure reasoning returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise AzureReasoningResponseError("Azure reasoning response must be an object")
    return payload


def _deployment_names(payload: dict) -> list[str]:
    deployments = payload.get("data")
    if not isinstance(deployments, list):
        raise AzureReasoningResponseError(
            "Azure deployment-list response was invalid"
        )
    return sorted({
        str(item["id"])
        for item in deployments
        if isinstance(item, dict) and item.get("id")
    })


def _azure_http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        response_bytes = error.read()
        payload = json.loads(response_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    error_payload = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error_payload, dict):
        return ""
    code = str(error_payload.get("code", "")).strip()
    message = str(error_payload.get("message", "")).strip()
    bounded_message = message[:MAXIMUM_AZURE_ERROR_MESSAGE_LENGTH]
    details = ": ".join(part for part in (code, bounded_message) if part)
    return f" ({details})" if details else ""


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
