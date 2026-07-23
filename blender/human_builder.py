import bpy

from materials import confidence_color, create_material


def build_human(entity: dict) -> dict:
    height_scale = entity["body_proportions"]["height_scale"]
    alpha = 1.0 if entity["fidelity_tier"] == "supported" else 0.62
    if entity["fidelity_tier"] == "weak":
        return _build_silhouette(entity, height_scale)
    materials = _human_materials(entity, alpha)
    root = _empty(f"Human_{entity['id']}", None, (0.0, 0.0, 0.0))
    _body_core(root, height_scale, materials)
    arms = [
        _arm(root, -1.0, height_scale, materials),
        _arm(root, 1.0, height_scale, materials),
    ]
    legs = [
        _leg(root, -1.0, height_scale, materials),
        _leg(root, 1.0, height_scale, materials),
    ]
    return {
        "root": root,
        "arms": arms,
        "legs": legs,
        "wheels": [],
        "steering_wheels": [],
        "wheel_radius": None,
        "materials": list(materials.values()),
        "visual_height_meters": 1.98 * height_scale,
    }


def _human_materials(entity: dict, alpha: float) -> dict:
    return {
        "upper": create_material(f"Upper_{entity['id']}", entity["appearance"]["upper_color"], alpha=alpha),
        "lower": create_material(f"Lower_{entity['id']}", entity["appearance"]["lower_color"], alpha=alpha),
        "skin": create_material(f"Neutral_{entity['id']}", (0.48, 0.37, 0.30), alpha=alpha),
        "shoe": create_material(f"Shoe_{entity['id']}", (0.025, 0.035, 0.045), alpha=alpha),
    }


def _body_core(root: bpy.types.Object, scale: float, materials: dict) -> None:
    _cube("Pelvis", (0.0, 0.0, 0.86 * scale), (0.24, 0.15, 0.15), materials["lower"], root)
    _cube("Torso", (0.0, 0.0, 1.25 * scale), (0.29, 0.17, 0.38), materials["upper"], root)
    _cylinder("Neck", (0.0, 0.0, 1.58 * scale), 0.07, 0.13, materials["skin"], root)
    _sphere("Head", (0.0, 0.0, 1.79 * scale), 0.19, materials["skin"], root)


def _arm(
    root: bpy.types.Object, side: float, scale: float, materials: dict,
) -> bpy.types.Object:
    shoulder = _empty("Shoulder", root, (side * 0.36, 0.0, 1.46 * scale))
    _cylinder("UpperArm", (0.0, 0.0, -0.22), 0.075, 0.44, materials["upper"], shoulder)
    _cylinder("Forearm", (0.0, 0.015, -0.59), 0.062, 0.34, materials["skin"], shoulder)
    _sphere("Hand", (0.0, 0.02, -0.80), 0.075, materials["skin"], shoulder)
    return shoulder


def _leg(
    root: bpy.types.Object, side: float, scale: float, materials: dict,
) -> bpy.types.Object:
    hip = _empty("Hip", root, (side * 0.14, 0.0, 0.96 * scale))
    _cylinder(
        "Thigh", (0.0, 0.0, -0.23 * scale),
        0.105 * scale, 0.46 * scale, materials["lower"], hip,
    )
    _cylinder(
        "Shin", (0.0, 0.015, -0.65 * scale),
        0.085 * scale, 0.40 * scale, materials["lower"], hip,
    )
    _cube(
        "Foot", (0.0, -0.07, -0.89 * scale),
        (0.10 * scale, 0.20 * scale, 0.07 * scale),
        materials["shoe"], hip,
    )
    return hip


def _build_silhouette(entity: dict, height_scale: float) -> dict:
    material = create_material(
        f"Weak_{entity['id']}", confidence_color(entity["confidence"]), alpha=0.40
    )
    root = _empty(f"Human_{entity['id']}", None, (0.0, 0.0, 0.0))
    bpy.ops.mesh.primitive_cone_add(
        vertices=12,
        radius1=0.22 * height_scale,
        radius2=0.14 * height_scale,
        depth=1.28 * height_scale,
    )
    presence = bpy.context.object
    presence.name = "UncertainPresence"
    presence.location = (0.0, 0.0, 0.78 * height_scale)
    presence.data.materials.append(material)
    presence.parent = root
    _sphere("UncertainHead", (0.0, 0.0, 1.75 * height_scale), 0.17, material, root)
    return {
        "root": root,
        "arms": [],
        "legs": [],
        "wheels": [],
        "steering_wheels": [],
        "wheel_radius": None,
        "materials": [material],
        "visual_height_meters": 1.92 * height_scale,
    }


def _empty(
    name: str, parent: bpy.types.Object | None, location: tuple[float, float, float],
) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    empty.parent = parent
    empty.location = location
    return empty


def _cube(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    part = bpy.context.object
    part.name = name
    part.location = location
    part.scale = scale
    part.data.materials.append(material)
    part.parent = parent
    return part

def _sphere(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius)
    part = bpy.context.object
    part.name = name
    part.location = location
    part.data.materials.append(material)
    part.parent = parent
    return part


def _cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=depth)
    part = bpy.context.object
    part.name = name
    part.location = location
    part.data.materials.append(material)
    part.parent = parent
    return part
