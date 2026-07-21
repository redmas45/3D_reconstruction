import math


MAX_BLENDER_FPS = 32_767


def blender_frame_rate(source_fps: float) -> tuple[int, float]:
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("Source frame rate must be a positive finite number")
    nominal_fps = min(MAX_BLENDER_FPS, max(1, math.floor(source_fps + 0.5)))
    return nominal_fps, nominal_fps / source_fps
