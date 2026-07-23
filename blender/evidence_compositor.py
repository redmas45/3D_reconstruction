from pathlib import Path

import bpy


PRESENTATION_NODE_PREFIX = "ReconstructPresentation"


def configure_evidence_compositor(scene: bpy.types.Scene, plan: dict) -> bool:
    environment = plan.get("environment", {})
    context_path = environment.get("context_frame_path")
    if not environment.get("hybrid_backplate_enabled") or not context_path:
        return False
    image_path = Path(str(context_path))
    if not image_path.is_file():
        return False
    scene.render.film_transparent = True
    scene.use_nodes = True
    node_tree = scene.node_tree
    _clear_nodes(node_tree)
    render_layers = _new_node(node_tree, "CompositorNodeRLayers", "RenderLayers")
    backplate = _new_node(node_tree, "CompositorNodeImage", "VisibleEvidence")
    backplate.image = bpy.data.images.load(str(image_path), check_existing=True)
    scale = _new_node(node_tree, "CompositorNodeScale", "Scale")
    scale.space = "RENDER_SIZE"
    scale.frame_method = "CROP"
    grade = _new_node(node_tree, "CompositorNodeHueSat", "EvidenceGrade")
    grade.inputs["Saturation"].default_value = 0.78
    grade.inputs["Value"].default_value = 0.74
    alpha_over = _new_node(node_tree, "CompositorNodeAlphaOver", "AlphaOver")
    alpha_over.inputs[0].default_value = 1.0
    composite = _new_node(node_tree, "CompositorNodeComposite", "Composite")
    node_tree.links.new(backplate.outputs["Image"], scale.inputs["Image"])
    node_tree.links.new(scale.outputs["Image"], grade.inputs["Image"])
    node_tree.links.new(grade.outputs["Image"], alpha_over.inputs[1])
    node_tree.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    node_tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])
    return True


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
