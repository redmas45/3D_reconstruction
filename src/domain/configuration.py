import json
from pathlib import Path


REQUIRED_CONFIGURATION_SECTIONS = (
    "yolo", "gap", "scene", "reasoning", "visualization", "evaluation", "renderer",
)
MINIMUM_GAP_SECONDS = 0.1
MAXIMUM_MISSING_FRACTION = 0.95
MAXIMUM_PARALLEL_GAP_RENDERERS = 4
MINIMUM_RENDER_STALL_TIMEOUT_SECONDS = 60
MAXIMUM_RENDER_STALL_TIMEOUT_SECONDS = 86_400
SUPPORTED_BLENDER_ENGINES = frozenset({"BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH", "CYCLES"})
SUPPORTED_CYCLES_COMPUTE_DEVICES = frozenset({"CUDA", "OPTIX"})
MAXIMUM_CYCLES_SAMPLES = 4_096
MINIMUM_RENDER_RUNTIME_BUDGET_SECONDS = 60
MAXIMUM_RENDER_RUNTIME_BUDGET_SECONDS = 21_600
MINIMUM_REASONING_TIMEOUT_SECONDS = 10
MAXIMUM_REASONING_TIMEOUT_SECONDS = 600
MINIMUM_REASONING_OUTPUT_TOKENS = 512
MAXIMUM_REASONING_OUTPUT_TOKENS = 32_000
SUPPORTED_REASONING_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh"})
SUPPORTED_PRODUCTION_HUD_MODES = frozenset({"minimal", "technical"})


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
    _validate_reasoning_configuration(configuration["reasoning"])
    _validate_renderer_configuration(configuration["renderer"])


def _validate_gap_configuration(gap_configuration: dict) -> None:
    missing_fraction = _required_number(gap_configuration, "missing_fraction")
    minimum_seconds = _required_number(gap_configuration, "min_seconds")
    maximum_seconds = _required_number(gap_configuration, "max_seconds")
    compact_minimum_seconds = _required_number(gap_configuration, "compact_min_seconds")
    compact_maximum_seconds = _required_number(gap_configuration, "compact_max_seconds")
    review_minimum_video_seconds = _required_number(
        gap_configuration, "review_profile_min_video_seconds",
    )
    if not 0.0 < missing_fraction <= MAXIMUM_MISSING_FRACTION:
        raise ConfigurationValidationError("gap.missing_fraction must be greater than 0 and at most 0.95")
    if minimum_seconds < MINIMUM_GAP_SECONDS or maximum_seconds < minimum_seconds:
        raise ConfigurationValidationError("Gap duration bounds are invalid")
    if (
        compact_minimum_seconds < MINIMUM_GAP_SECONDS
        or compact_maximum_seconds < compact_minimum_seconds
    ):
        raise ConfigurationValidationError("Compact gap duration bounds are invalid")
    if review_minimum_video_seconds <= 0:
        raise ConfigurationValidationError("gap.review_profile_min_video_seconds must be positive")


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
    _validate_smart_renderer_configuration(renderer_configuration)
    _validate_optional_cycles_configuration(renderer_configuration)


def _validate_reasoning_configuration(reasoning_configuration: dict) -> None:
    if not isinstance(reasoning_configuration.get("enabled"), bool):
        raise ConfigurationValidationError("reasoning.enabled must be boolean")
    timeout_seconds = _required_integer(reasoning_configuration, "request_timeout_seconds")
    if not MINIMUM_REASONING_TIMEOUT_SECONDS <= timeout_seconds <= MAXIMUM_REASONING_TIMEOUT_SECONDS:
        raise ConfigurationValidationError("reasoning.request_timeout_seconds must be between 10 and 600")
    output_tokens = _required_integer(reasoning_configuration, "max_output_tokens")
    if not MINIMUM_REASONING_OUTPUT_TOKENS <= output_tokens <= MAXIMUM_REASONING_OUTPUT_TOKENS:
        raise ConfigurationValidationError("reasoning.max_output_tokens must be between 512 and 32000")
    if reasoning_configuration.get("reasoning_effort") not in SUPPORTED_REASONING_EFFORTS:
        raise ConfigurationValidationError("reasoning.reasoning_effort is unsupported")
    if _required_integer(reasoning_configuration, "planner_schema_version") != 2:
        raise ConfigurationValidationError("reasoning.planner_schema_version must be 2")
    if not 1 <= _required_integer(reasoning_configuration, "maximum_gaps_per_batch") <= 16:
        raise ConfigurationValidationError("reasoning.maximum_gaps_per_batch must be between 1 and 16")
    if not 1 <= _required_integer(reasoning_configuration, "maximum_images_per_batch") <= 64:
        raise ConfigurationValidationError("reasoning.maximum_images_per_batch must be between 1 and 64")
    if reasoning_configuration.get("image_detail") not in {"low", "high", "original", "auto"}:
        raise ConfigurationValidationError("reasoning.image_detail is unsupported")
    _validate_reasoning_images(reasoning_configuration.get("images"))


def _validate_reasoning_images(value: object) -> None:
    if not isinstance(value, dict):
        raise ConfigurationValidationError("reasoning.images must be an object")
    for field_name in ("max_global_keyframes", "boundary_frames_per_side", "crops_per_track"):
        field_value = _required_integer(value, field_name)
        if not 1 <= field_value <= 32:
            raise ConfigurationValidationError(f"reasoning.images.{field_name} must be between 1 and 32")


def _validate_smart_renderer_configuration(configuration: dict) -> None:
    bounded_fields = {
        "target_fps": (1, 60),
        "scale_percent": (1, 100),
        "maximum_detailed_entities": (1, 64),
        "maximum_gpu_workers": (1, 1),
        "maximum_cpu_workers": (1, 16),
        "checkpoint_frame_batch": (1, 500),
        "diagnostic_pose_count": (1, 16),
        "minimum_render_long_edge": (320, 7680),
        "maximum_render_long_edge": (320, 7680),
    }
    for field_name, bounds in bounded_fields.items():
        value = _required_integer(configuration, field_name)
        if not bounds[0] <= value <= bounds[1]:
            raise ConfigurationValidationError(
                f"renderer.{field_name} must be between {bounds[0]} and {bounds[1]}"
            )
    if not isinstance(configuration.get("requires_preview_approval"), bool):
        raise ConfigurationValidationError("renderer.requires_preview_approval must be boolean")
    if not isinstance(configuration.get("hybrid_static_backplate"), bool):
        raise ConfigurationValidationError("renderer.hybrid_static_backplate must be boolean")
    if configuration.get("production_hud_mode") not in SUPPORTED_PRODUCTION_HUD_MODES:
        raise ConfigurationValidationError("renderer.production_hud_mode is unsupported")
    if (
        int(configuration["maximum_render_long_edge"])
        < int(configuration["minimum_render_long_edge"])
    ):
        raise ConfigurationValidationError(
            "renderer.maximum_render_long_edge must not be smaller than the minimum"
        )
    for field_name in (
        "runtime_budget_enabled",
        "allow_runtime_budget_override",
        "interactive_preview_approval",
    ):
        if not isinstance(configuration.get(field_name), bool):
            raise ConfigurationValidationError(f"renderer.{field_name} must be boolean")
    maximum_runtime = _required_integer(configuration, "maximum_predicted_render_seconds")
    if not MINIMUM_RENDER_RUNTIME_BUDGET_SECONDS <= maximum_runtime <= MAXIMUM_RENDER_RUNTIME_BUDGET_SECONDS:
        raise ConfigurationValidationError(
            "renderer.maximum_predicted_render_seconds must be between 60 and 21600"
        )


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
