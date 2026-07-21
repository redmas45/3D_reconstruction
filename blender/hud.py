from pathlib import Path

import bpy

from materials import create_material


HUD_DEPTH = -3.0
OVERLAY_SCALE = 0.375


def build_hud(plan: dict, camera: bpy.types.Object) -> None:
    _build_header(plan, camera)
    context_frame_path = plan["environment"].get("context_frame_path")
    if context_frame_path and Path(context_frame_path).is_file():
        _build_evidence_panel(Path(context_frame_path), camera)
    _build_mode_transition(plan, camera)


def _build_header(plan: dict, camera: bpy.types.Object) -> None:
    confidence = plan["overall_confidence"]
    calibration = plan["camera"]["calibration_confidence"]
    title = _camera_text("AI-INFERRED FORENSIC 3D", camera, _overlay_location(-3.45, 1.85), 0.25 * OVERLAY_SCALE)
    details = _camera_text(
        f"GAP {plan['gap_index'] + 1:02d}  |  {plan['duration_seconds']:.2f}s  |  {len(plan['entities'])} ENTITIES",
        camera,
        _overlay_location(-3.45, 1.54),
        0.16 * OVERLAY_SCALE,
    )
    confidence_text = _camera_text(
        f"SCENE {confidence:.0%}  |  CAMERA PRIOR {calibration:.0%}  |  INFERRED, NOT GROUND TRUTH",
        camera,
        _overlay_location(-3.45, -1.86),
        0.115 * OVERLAY_SCALE,
    )
    for index, text_object in enumerate((title, details, confidence_text)):
        text_object.data.materials.append(
            create_material(f"HUD_Text_Material_{index}", (0.72, 0.95, 0.98), alpha=0.94)
        )


def _build_evidence_panel(image_path: Path, camera: bpy.types.Object) -> None:
    border = _camera_plane(camera, _overlay_location(-3.0, 0.75, -0.004), (0.67 * OVERLAY_SCALE, 0.42 * OVERLAY_SCALE))
    border.data.materials.append(create_material("EvidencePanelBorder", (0.03, 0.68, 0.72), alpha=0.80))
    panel = _camera_plane(camera, _overlay_location(-3.0, 0.75), (0.62 * OVERLAY_SCALE, 0.35 * OVERLAY_SCALE))
    panel.data.materials.append(_image_material(image_path))
    label = _camera_text(
        "LAST VISIBLE EVIDENCE", camera, _overlay_location(-3.62, 1.21), 0.12 * OVERLAY_SCALE
    )
    label.data.materials.append(create_material("EvidencePanelLabel", (0.70, 0.94, 0.97), alpha=0.92))


def _image_material(image_path: Path) -> bpy.types.Material:
    material = bpy.data.materials.new("VisibleEvidenceFrame")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(image_path), check_existing=True)
    emission.inputs["Strength"].default_value = 0.72
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _build_mode_transition(plan: dict, camera: bpy.types.Object) -> None:
    final_frame = int(plan["frame_count"])
    shutter = _camera_plane(camera, (0.0, 0.0, -2.50), (1.45, 0.84))
    shutter_material = create_material("Inference_Shutter", (0.005, 0.012, 0.020), alpha=1.0)
    shutter.data.materials.append(shutter_material)
    _keyframe_alpha(shutter_material, [(1, 1.0), (2, 1.0), (6, 0.0), (final_frame - 5, 0.0), (final_frame, 1.0)])
    entering = _camera_text("ENTERING AI INFERENCE", camera, (-0.82, 0.0, -2.42), 0.105)
    returning = _camera_text("RETURNING TO VISIBLE EVIDENCE", camera, (-1.12, 0.0, -2.41), 0.085)
    entering_material = create_material("Entering_Inference_Text", (0.25, 0.95, 0.98), alpha=1.0)
    returning_material = create_material("Returning_Evidence_Text", (0.25, 0.95, 0.98), alpha=0.0)
    entering.data.materials.append(entering_material)
    returning.data.materials.append(returning_material)
    _keyframe_alpha(entering_material, [(1, 1.0), (5, 1.0), (7, 0.0)])
    _keyframe_alpha(returning_material, [(1, 0.0), (final_frame - 5, 0.0), (final_frame, 1.0)])


def _keyframe_alpha(material: bpy.types.Material, keyframes: list[tuple[int, float]]) -> None:
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader is None:
        return
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    alpha_input = shader.inputs["Alpha"]
    for frame, alpha in keyframes:
        alpha_input.default_value = alpha
        alpha_input.keyframe_insert("default_value", frame=frame)


def _camera_plane(
    camera: bpy.types.Object,
    location: tuple[float, float, float],
    scale: tuple[float, float],
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    plane = bpy.context.object
    plane.parent = camera
    plane.location = location
    plane.rotation_euler = (0.0, 0.0, 0.0)
    plane.scale = (scale[0], scale[1], 1.0)
    return plane


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
    text_data.extrude = 0.0
    text_object = bpy.data.objects.new("HUD_Text", text_data)
    bpy.context.collection.objects.link(text_object)
    text_object.parent = camera
    text_object.location = location
    if hasattr(text_object, "visible_shadow"):
        text_object.visible_shadow = False
    return text_object


def _overlay_location(x_position: float, y_position: float, depth_offset: float = 0.0) -> tuple[float, float, float]:
    return (x_position * OVERLAY_SCALE, y_position * OVERLAY_SCALE, HUD_DEPTH + depth_offset)
