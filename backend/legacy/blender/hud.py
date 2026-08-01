import bpy

from materials import create_material


HUD_DEPTH = -3.0
OVERLAY_SCALE = 0.375


def build_hud(plan: dict, camera: bpy.types.Object) -> None:
    mode = plan.get("render", {}).get("production_hud_mode", "minimal")
    if mode == "technical":
        _build_technical_badge(plan, camera)
        return
    _build_minimal_badge(plan, camera)


def _build_minimal_badge(plan: dict, camera: bpy.types.Object) -> None:
    badge = _camera_text(
        "AI RECONSTRUCTION",
        camera,
        _overlay_location(-3.48, 1.82),
        0.20 * OVERLAY_SCALE,
    )
    badge.data.materials.append(
        create_material("HUD_Badge", (0.72, 0.95, 0.98), alpha=0.90),
    )


def _build_technical_badge(plan: dict, camera: bpy.types.Object) -> None:
    confidence = float(plan["overall_confidence"])
    calibration = float(plan["camera"]["calibration_confidence"])
    content = (
        f"AI RECONSTRUCTION | GAP {int(plan['gap_index']) + 1:02d} | "
        f"SCENE {confidence:.0%} | CAMERA {calibration:.0%}"
    )
    badge = _camera_text(
        content,
        camera,
        _overlay_location(-3.48, 1.82),
        0.17 * OVERLAY_SCALE,
    )
    badge.data.materials.append(
        create_material("HUD_Technical_Badge", (0.72, 0.95, 0.98), alpha=0.90),
    )


def _camera_text(
    content: str,
    camera: bpy.types.Object,
    location: tuple[float, float, float],
    size: float,
) -> bpy.types.Object:
    text_data = bpy.data.curves.new(type="FONT", name="HUD_Text")
    text_data.body = content
    text_data.align_x = "LEFT"
    text_data.size = size
    text_object = bpy.data.objects.new("HUD_Text", text_data)
    bpy.context.collection.objects.link(text_object)
    text_object.parent = camera
    text_object.location = location
    if hasattr(text_object, "visible_shadow"):
        text_object.visible_shadow = False
    return text_object


def _overlay_location(
    x_position: float,
    y_position: float,
) -> tuple[float, float, float]:
    return (
        x_position * OVERLAY_SCALE,
        y_position * OVERLAY_SCALE,
        HUD_DEPTH,
    )
