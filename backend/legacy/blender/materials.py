import bpy


def create_material(
    name: str,
    color: tuple[float, float, float] | list[float],
    alpha: float = 1.0,
    metallic: float = 0.0,
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.diffuse_color = (*color, alpha)
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Alpha"].default_value = alpha
    if alpha < 1.0:
        material.diffuse_color = (*color, alpha)
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
    return material


def confidence_color(confidence: float) -> tuple[float, float, float]:
    if confidence >= 0.75:
        return (0.08, 0.82, 0.58)
    if confidence >= 0.50:
        return (0.96, 0.61, 0.08)
    return (0.94, 0.24, 0.27)
