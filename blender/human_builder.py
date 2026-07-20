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
    ring_material = _add_confidence_ring(entity, root)
    return {
        "root": root,
        "arms": arms,
        "legs": legs,
        "wheels": [],
        "materials": [*materials.values(), ring_material],
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
    hip = _empty("Hip", root, (side * 0.14, 0.0, 0.82 * scale))
    _cylinder("Thigh", (0.0, 0.0, -0.23), 0.105, 0.46, materials["lower"], hip)
    _cylinder("Shin", (0.0, 0.015, -0.65), 0.085, 0.40, materials["lower"], hip)
    _cube("Foot", (0.0, -0.07, -0.89), (0.10, 0.20, 0.07), materials["shoe"], hip)
    return hip


def _build_silhouette(entity: dict, height_scale: float) -> dict:
    material = create_material(
        f"Weak_{entity['id']}", confidence_color(entity["confidence"]), alpha=0.14
    )
    root = _empty(f"Human_{entity['id']}", None, (0.0, 0.0, 0.0))
    _cylinder("UncertainPresence", (0.0, 0.0, 0.90 * height_scale), 0.18, 1.42, material, root)
    _sphere("UncertainHead", (0.0, 0.0, 1.75 * height_scale), 0.17, material, root)
    ring_material = _add_confidence_ring(entity, root)
    return {
        "root": root,
        "arms": [],
        "legs": [],
        "wheels": [],
        "materials": [material, ring_material],
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


def _add_confidence_ring(entity: dict, root: bpy.types.Object) -> bpy.types.Material:
    radius = 0.48 + entity["uncertainty"]["position_radius_meters"] * 0.16
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=0.018, major_segments=32)
    ring = bpy.context.object
    ring.name = f"Confidence_{entity['id']}"
    ring.location = (0.0, 0.0, 0.025)
    ring_material = create_material(
        f"ConfidenceMaterial_{entity['id']}", confidence_color(entity["confidence"]), alpha=0.58
    )
    ring.data.materials.append(ring_material)
    ring.parent = root
    return ring_material
