"""Extracts bounded diagnostic layers from the same Blender render."""

from __future__ import annotations

from pathlib import Path

import bpy


ENVIRONMENT_PASS_INDEX = 1
ACTOR_PASS_INDEX = 2
UNCERTAINTY_PASS_INDEX = 3
HUD_PASS_INDEX = 4
DIAGNOSTIC_NODE_PREFIX = "ReconstructDiagnostic"
WORKBENCH_RENDER_ENGINE = "BLENDER_WORKBENCH"


def assign_pass_index(objects: set[bpy.types.Object], pass_index: int) -> None:
    for scene_object in objects:
        if scene_object.type != "EMPTY":
            scene_object.pass_index = pass_index


def configure_diagnostic_passes(
    scene: bpy.types.Scene,
    output_directory: Path,
) -> dict:
    if scene.render.engine == WORKBENCH_RENDER_ENGINE:
        return {
            "available": False,
            "reason": "Workbench does not expose the required object-index render passes",
        }
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True
    if hasattr(view_layer, "use_pass_shadow"):
        view_layer.use_pass_shadow = True
    scene.use_nodes = True
    node_tree = scene.node_tree
    _remove_diagnostic_nodes(node_tree)
    render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = f"{DIAGNOSTIC_NODE_PREFIX}_RenderLayers"
    composite = _presentation_composite(node_tree)
    if composite is None:
        composite = node_tree.nodes.new("CompositorNodeComposite")
        composite.name = f"{DIAGNOSTIC_NODE_PREFIX}_Composite"
        node_tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])
    diagnostic_root = output_directory / "diagnostic_layers"
    _add_index_mask(node_tree, render_layers, diagnostic_root, "environment", ENVIRONMENT_PASS_INDEX)
    _add_index_mask(node_tree, render_layers, diagnostic_root, "actors", ACTOR_PASS_INDEX)
    _add_index_mask(node_tree, render_layers, diagnostic_root, "uncertainty", UNCERTAINTY_PASS_INDEX)
    _add_index_mask(node_tree, render_layers, diagnostic_root, "hud", HUD_PASS_INDEX)
    _add_depth_output(node_tree, render_layers, diagnostic_root)
    _add_shadow_output(node_tree, render_layers, diagnostic_root)
    return {
        "available": True,
        "strategy": "single_render_compositor_passes",
        "layers": ["environment", "actors", "uncertainty", "hud", "depth", "shadow"],
        "directory": str(diagnostic_root),
    }


def _presentation_composite(node_tree: bpy.types.NodeTree) -> bpy.types.Node | None:
    for node in node_tree.nodes:
        if (
            node.bl_idname == "CompositorNodeComposite"
            and node.name.startswith("ReconstructPresentation")
        ):
            return node
    return None


def set_diagnostic_output_enabled(scene: bpy.types.Scene, enabled: bool) -> None:
    if not scene.use_nodes or scene.node_tree is None:
        return
    for node in scene.node_tree.nodes:
        if node.name.startswith(f"{DIAGNOSTIC_NODE_PREFIX}_Output_"):
            node.mute = not enabled


def diagnostic_layer_files(output_directory: Path) -> dict[str, list[str]]:
    root = output_directory / "diagnostic_layers"
    if not root.is_dir():
        return {}
    return {
        layer_directory.name: [
            str(path)
            for path in sorted(layer_directory.glob("*.png"))
            if path.is_file()
        ]
        for layer_directory in sorted(root.iterdir())
        if layer_directory.is_dir()
    }


def _remove_diagnostic_nodes(node_tree: bpy.types.NodeTree) -> None:
    for node in list(node_tree.nodes):
        if node.name.startswith(DIAGNOSTIC_NODE_PREFIX):
            node_tree.nodes.remove(node)


def _add_index_mask(
    node_tree: bpy.types.NodeTree,
    render_layers: bpy.types.Node,
    output_root: Path,
    name: str,
    pass_index: int,
) -> None:
    index_output = render_layers.outputs.get("IndexOB")
    if index_output is None:
        return
    mask = node_tree.nodes.new("CompositorNodeIDMask")
    mask.name = f"{DIAGNOSTIC_NODE_PREFIX}_Mask_{name}"
    mask.index = pass_index
    mask.use_antialiasing = True
    node_tree.links.new(index_output, mask.inputs["ID value"])
    _add_file_output(node_tree, mask.outputs["Alpha"], output_root, name)


def _add_depth_output(
    node_tree: bpy.types.NodeTree,
    render_layers: bpy.types.Node,
    output_root: Path,
) -> None:
    depth_output = render_layers.outputs.get("Depth")
    if depth_output is None:
        return
    normalize = node_tree.nodes.new("CompositorNodeNormalize")
    normalize.name = f"{DIAGNOSTIC_NODE_PREFIX}_Normalize_depth"
    node_tree.links.new(depth_output, normalize.inputs["Value"])
    _add_file_output(node_tree, normalize.outputs["Value"], output_root, "depth")


def _add_shadow_output(
    node_tree: bpy.types.NodeTree,
    render_layers: bpy.types.Node,
    output_root: Path,
) -> None:
    shadow_output = render_layers.outputs.get("Shadow")
    if shadow_output is not None:
        _add_file_output(node_tree, shadow_output, output_root, "shadow")


def _add_file_output(
    node_tree: bpy.types.NodeTree,
    source_socket: bpy.types.NodeSocket,
    output_root: Path,
    name: str,
) -> None:
    output = node_tree.nodes.new("CompositorNodeOutputFile")
    output.name = f"{DIAGNOSTIC_NODE_PREFIX}_Output_{name}"
    output.base_path = str(output_root)
    output.file_slots[0].path = f"{name}/{name}_"
    output.format.file_format = "PNG"
    output.format.color_mode = "BW"
    output.format.color_depth = "8"
    output.format.compression = 75
    node_tree.links.new(source_socket, output.inputs[0])
