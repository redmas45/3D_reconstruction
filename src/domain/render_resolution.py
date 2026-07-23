import math


DEFAULT_MINIMUM_RENDER_LONG_EDGE = 960
DEFAULT_MAXIMUM_RENDER_LONG_EDGE = 1280


def adaptive_render_scale_percent(
    source_width: int,
    source_height: int,
    configured_scale_percent: int,
    minimum_long_edge: int = DEFAULT_MINIMUM_RENDER_LONG_EDGE,
    maximum_long_edge: int = DEFAULT_MAXIMUM_RENDER_LONG_EDGE,
) -> int:
    source_long_edge = max(source_width, source_height)
    if source_long_edge <= 0:
        raise ValueError("Source dimensions must be positive")
    if minimum_long_edge <= 0 or maximum_long_edge < minimum_long_edge:
        raise ValueError("Adaptive render bounds are invalid")
    configured_long_edge = source_long_edge * configured_scale_percent / 100.0
    target_long_edge = min(
        float(source_long_edge),
        float(maximum_long_edge),
        max(float(minimum_long_edge), configured_long_edge),
    )
    return max(
        1,
        min(100, math.floor(target_long_edge / source_long_edge * 100.0)),
    )


def scaled_render_dimensions(
    source_width: int,
    source_height: int,
    scale_percent: int,
) -> tuple[int, int]:
    return (
        _even_dimension(round(source_width * scale_percent / 100.0)),
        _even_dimension(round(source_height * scale_percent / 100.0)),
    )


def _even_dimension(value: int) -> int:
    bounded_value = max(2, value)
    return bounded_value if bounded_value % 2 == 0 else bounded_value + 1
