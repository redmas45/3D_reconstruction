import json
from pathlib import Path


REQUIRED_CONFIGURATION_SECTIONS = ("yolo", "gap", "scene", "visualization", "evaluation", "renderer")
MINIMUM_GAP_SECONDS = 0.1
MAXIMUM_MISSING_FRACTION = 0.95
MAXIMUM_PARALLEL_GAP_RENDERERS = 4
MINIMUM_RENDER_STALL_TIMEOUT_SECONDS = 60
MAXIMUM_RENDER_STALL_TIMEOUT_SECONDS = 86_400
SUPPORTED_BLENDER_ENGINES = frozenset({"BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH", "CYCLES"})
SUPPORTED_CYCLES_COMPUTE_DEVICES = frozenset({"CUDA", "OPTIX"})
MAXIMUM_CYCLES_SAMPLES = 4_096


class ConfigurationValidationError(ValueError):
    pass


def load_validated_configuration(configuration_path: Path) -> dict:
    try:
        with configuration_path.open("r", encoding="utf-8") as configuration_file:
            payload = json.load(configuration_file)
    except json.JSONDecodeError as error:
        raise ConfigurationValidationError(f"Configuration contains invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ConfigurationValidationError("Configuration root must be an object")
    validate_configuration(payload)
    return payload


def validate_configuration(configuration: dict) -> None:
    for section_name in REQUIRED_CONFIGURATION_SECTIONS:
        if not isinstance(configuration.get(section_name), dict):
            raise ConfigurationValidationError(f"Configuration section '{section_name}' must be an object")
    _validate_gap_configuration(configuration["gap"])
    _validate_yolo_configuration(configuration["yolo"])
    _validate_renderer_configuration(configuration["renderer"])


def _validate_gap_configuration(gap_configuration: dict) -> None:
    missing_fraction = _required_number(gap_configuration, "missing_fraction")
    minimum_seconds = _required_number(gap_configuration, "min_seconds")
    maximum_seconds = _required_number(gap_configuration, "max_seconds")
    if not 0.0 < missing_fraction <= MAXIMUM_MISSING_FRACTION:
        raise ConfigurationValidationError("gap.missing_fraction must be greater than 0 and at most 0.95")
    if minimum_seconds < MINIMUM_GAP_SECONDS or maximum_seconds < minimum_seconds:
        raise ConfigurationValidationError("Gap duration bounds are invalid")


def _validate_yolo_configuration(yolo_configuration: dict) -> None:
    confidence = _required_number(yolo_configuration, "confidence")
    frame_stride = _required_integer(yolo_configuration, "frame_stride")
    if not 0.0 <= confidence <= 100.0:
        raise ConfigurationValidationError("yolo.confidence must be between 0 and 1, or a percentage up to 100")
    if frame_stride < 1:
        raise ConfigurationValidationError("yolo.frame_stride must be at least 1")


def _validate_renderer_configuration(renderer_configuration: dict) -> None:
    default_mode = renderer_configuration.get("default_mode")
    if default_mode not in {"blender", "2d"}:
        raise ConfigurationValidationError("renderer.default_mode must be 'blender' or '2d'")
    for field_name in ("preview_scale_percent", "production_scale_percent"):
        scale_percent = _required_integer(renderer_configuration, field_name)
        if not 1 <= scale_percent <= 100:
            raise ConfigurationValidationError(f"renderer.{field_name} must be between 1 and 100")
    parallel_renderers = _required_integer(renderer_configuration, "max_parallel_gap_renders")
    if not 1 <= parallel_renderers <= MAXIMUM_PARALLEL_GAP_RENDERERS:
        raise ConfigurationValidationError(
            f"renderer.max_parallel_gap_renders must be between 1 and {MAXIMUM_PARALLEL_GAP_RENDERERS}"
        )
    stall_timeout = _required_integer(renderer_configuration, "gap_render_stall_timeout_seconds")
    if not MINIMUM_RENDER_STALL_TIMEOUT_SECONDS <= stall_timeout <= MAXIMUM_RENDER_STALL_TIMEOUT_SECONDS:
        raise ConfigurationValidationError(
            "renderer.gap_render_stall_timeout_seconds must be between 60 and 86400"
        )
    _validate_optional_cycles_configuration(renderer_configuration)


def _validate_optional_cycles_configuration(renderer_configuration: dict) -> None:
    engine = renderer_configuration.get("engine")
    if engine is not None and engine not in SUPPORTED_BLENDER_ENGINES:
        raise ConfigurationValidationError("renderer.engine is unsupported")
    compute_device = renderer_configuration.get("cycles_compute_device")
    if compute_device is not None and compute_device not in SUPPORTED_CYCLES_COMPUTE_DEVICES:
        raise ConfigurationValidationError("renderer.cycles_compute_device must be CUDA or OPTIX")
    samples = renderer_configuration.get("cycles_samples")
    if samples is not None and (
        isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAXIMUM_CYCLES_SAMPLES
    ):
        raise ConfigurationValidationError(
            f"renderer.cycles_samples must be between 1 and {MAXIMUM_CYCLES_SAMPLES}"
        )
    use_denoising = renderer_configuration.get("cycles_use_denoising")
    if use_denoising is not None and not isinstance(use_denoising, bool):
        raise ConfigurationValidationError("renderer.cycles_use_denoising must be boolean")


def _required_number(configuration: dict, field_name: str) -> float:
    value = configuration.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationValidationError(f"Configuration value '{field_name}' must be numeric")
    return float(value)


def _required_integer(configuration: dict, field_name: str) -> int:
    value = configuration.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationValidationError(f"Configuration value '{field_name}' must be an integer")
    return value
