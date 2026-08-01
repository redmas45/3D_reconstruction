from pathlib import Path

import bpy


PRESENTATION_NODE_PREFIX = "ReconstructPresentation"
CONTEXT_BLUR_PIXELS = 2
CONTEXT_SATURATION = 0.72
CONTEXT_VALUE = 0.70


def configure_evidence_compositor(scene: bpy.types.Scene, plan: dict) -> bool:
    environment = plan.get("environment", {})
    before_path = _existing_image(environment.get("context_frame_path"))
    if not environment.get("hybrid_backplate_enabled") or before_path is None:
        return False
    after_path = _existing_image(environment.get("post_context_frame_path"))
    scene.render.film_transparent = True
    scene.use_nodes = True
    node_tree = scene.node_tree
    _clear_nodes(node_tree)
    render_layers = _new_node(node_tree, "CompositorNodeRLayers", "RenderLayers")
    before_output = _context_output(node_tree, before_path, "Before")
    context_output = (
        _transition_output(node_tree, before_output, after_path, plan)
        if after_path is not None else before_output
    )
    graded_output = _grade_context(node_tree, context_output)
    _composite_actors(node_tree, graded_output, render_layers.outputs["Image"])
    return True


def _existing_image(value: object) -> Path | None:
    if not value:
        return None
    image_path = Path(str(value))
    return image_path if image_path.is_file() else None


def _context_output(
    node_tree: bpy.types.NodeTree,
    image_path: Path,
    suffix: str,
):
    image_node = _new_node(node_tree, "CompositorNodeImage", f"{suffix}Evidence")
    image_node.image = bpy.data.images.load(str(image_path), check_existing=True)
    scale = _new_node(node_tree, "CompositorNodeScale", f"{suffix}Scale")
    scale.space = "RENDER_SIZE"
    scale.frame_method = "CROP"
    blur = _new_node(node_tree, "CompositorNodeBlur", f"{suffix}Blur")
    blur.filter_type = "FAST_GAUSS"
    blur.size_x = CONTEXT_BLUR_PIXELS
    blur.size_y = CONTEXT_BLUR_PIXELS
    node_tree.links.new(image_node.outputs["Image"], scale.inputs["Image"])
    node_tree.links.new(scale.outputs["Image"], blur.inputs["Image"])
    return blur.outputs["Image"]


def _transition_output(
    node_tree: bpy.types.NodeTree,
    before_output,
    after_path: Path,
    plan: dict,
):
    after_output = _context_output(node_tree, after_path, "After")
    transition = _new_node(node_tree, "CompositorNodeMixRGB", "BoundaryTransition")
    transition.blend_type = "MIX"
    factor = transition.inputs[0]
    factor.default_value = 0.0
    factor.keyframe_insert("default_value", frame=1)
    factor.default_value = 1.0
    factor.keyframe_insert("default_value", frame=max(2, int(plan["frame_count"])))
    node_tree.links.new(before_output, transition.inputs[1])
    node_tree.links.new(after_output, transition.inputs[2])
    return transition.outputs["Image"]


def _grade_context(node_tree: bpy.types.NodeTree, context_output):
    grade = _new_node(node_tree, "CompositorNodeHueSat", "EvidenceGrade")
    grade.inputs["Saturation"].default_value = CONTEXT_SATURATION
    grade.inputs["Value"].default_value = CONTEXT_VALUE
    node_tree.links.new(context_output, grade.inputs["Image"])
    return grade.outputs["Image"]


def _composite_actors(node_tree: bpy.types.NodeTree, context_output, actor_output) -> None:
    alpha_over = _new_node(node_tree, "CompositorNodeAlphaOver", "AlphaOver")
    alpha_over.inputs[0].default_value = 1.0
    composite = _new_node(node_tree, "CompositorNodeComposite", "Composite")
    node_tree.links.new(context_output, alpha_over.inputs[1])
    node_tree.links.new(actor_output, alpha_over.inputs[2])
    node_tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])


def _new_node(
    node_tree: bpy.types.NodeTree,
    node_type: str,
    suffix: str,
) -> bpy.types.Node:
    node = node_tree.nodes.new(node_type)
    node.name = f"{PRESENTATION_NODE_PREFIX}_{suffix}"
    return node


def _clear_nodes(node_tree: bpy.types.NodeTree) -> None:
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)
