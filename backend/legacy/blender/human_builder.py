import hashlib

import bpy

from materials import create_material


IDENTITY_ACCENT_PALETTE = (
    (0.10, 0.78, 0.88),
    (0.95, 0.50, 0.18),
    (0.62, 0.45, 0.92),
    (0.18, 0.78, 0.48),
    (0.92, 0.34, 0.52),
)
MINIMUM_READABLE_ALPHA = 0.96
LOW_CHROMA_THRESHOLD = 0.16
ACCENT_BLEND_STRENGTH = 0.66
CONTACT_SHADOW_ALPHA = 0.42


def build_human(entity: dict) -> dict:
    height_scale = entity["body_proportions"]["height_scale"]
    alpha = 1.0 if entity["fidelity_tier"] == "supported" else MINIMUM_READABLE_ALPHA
    if entity["fidelity_tier"] == "weak":
        return _build_silhouette(entity, height_scale)
    materials = _human_materials(entity, alpha)
    root = _empty(f"Human_{entity['id']}", None, (0.0, 0.0, 0.0))
    body_rig = _empty("BodyRig", root, (0.0, 0.0, 0.0))
    _body_core(body_rig, height_scale, materials)
    arm_controls = [
        _arm(body_rig, -1.0, height_scale, materials),
        _arm(body_rig, 1.0, height_scale, materials),
    ]
    leg_controls = [
        _leg(root, -1.0, height_scale, materials),
        _leg(root, 1.0, height_scale, materials),
    ]
    shadow_material = create_contact_shadow(root, height_scale, str(entity["id"]))
    return {
        "root": root,
        "rig": body_rig,
        "arms": [control["shoulder"] for control in arm_controls],
        "elbows": [control["elbow"] for control in arm_controls],
        "legs": [control["hip"] for control in leg_controls],
        "knees": [control["knee"] for control in leg_controls],
        "feet": [control["foot"] for control in leg_controls],
        "leg_base_heights": [
            float(control["hip"].location.z)
            for control in leg_controls
        ],
        "height_scale": height_scale,
        "wheels": [],
        "steering_wheels": [],
        "wheel_radius": None,
        "materials": [*materials.values(), shadow_material],
        "visual_height_meters": 1.98 * height_scale,
    }


def _human_materials(entity: dict, alpha: float) -> dict:
    colors = human_colors(entity)
    return {
        "upper": create_material(
            f"Upper_{entity['id']}",
            colors["upper"],
            alpha=alpha,
        ),
        "lower": create_material(
            f"Lower_{entity['id']}",
            colors["lower"],
            alpha=alpha,
        ),
        "skin": create_material(f"Neutral_{entity['id']}", colors["skin"], alpha=alpha),
        "shoe": create_material(f"Shoe_{entity['id']}", colors["shoe"], alpha=alpha),
    }


def human_colors(entity: dict) -> dict[str, tuple[float, float, float] | list[float]]:
    accent = _identity_accent(str(entity["id"]))
    return {
        "upper": _readable_color(entity["appearance"]["upper_color"], accent),
        "lower": _readable_color(entity["appearance"]["lower_color"], accent),
        "skin": (0.48, 0.37, 0.30),
        "shoe": (0.025, 0.035, 0.045),
    }


def _body_core(root: bpy.types.Object, scale: float, materials: dict) -> None:
    pelvis = _cube(
        "Pelvis",
        (0.0, 0.0, 0.86 * scale),
        (0.24, 0.15, 0.15),
        materials["lower"],
        root,
    )
    torso = _cube(
        "Torso",
        (0.0, 0.0, 1.25 * scale),
        (0.29, 0.17, 0.38),
        materials["upper"],
        root,
    )
    _soften_mesh(pelvis, 0.04 * scale)
    _soften_mesh(torso, 0.08 * scale)
    _cylinder("Neck", (0.0, 0.0, 1.58 * scale), 0.07, 0.13, materials["skin"], root)
    _sphere("Head", (0.0, 0.0, 1.79 * scale), 0.19, materials["skin"], root)


def _arm(
    root: bpy.types.Object, side: float, scale: float, materials: dict,
) -> dict[str, bpy.types.Object]:
    shoulder = _empty("Shoulder", root, (side * 0.36, 0.0, 1.46 * scale))
    _cylinder(
        "UpperArm",
        (0.0, 0.0, -0.22 * scale),
        0.075 * scale,
        0.44 * scale,
        materials["upper"],
        shoulder,
    )
    elbow = _empty("Elbow", shoulder, (0.0, 0.0, -0.44 * scale))
    _cylinder(
        "Forearm",
        (0.0, 0.015, -0.17 * scale),
        0.062 * scale,
        0.34 * scale,
        materials["skin"],
        elbow,
    )
    _sphere(
        "Hand",
        (0.0, 0.02, -0.39 * scale),
        0.075 * scale,
        materials["skin"],
        elbow,
    )
    return {"shoulder": shoulder, "elbow": elbow}


def _leg(
    root: bpy.types.Object, side: float, scale: float, materials: dict,
) -> dict[str, bpy.types.Object]:
    hip = _empty("Hip", root, (side * 0.14, 0.0, 0.96 * scale))
    _cylinder(
        "Thigh", (0.0, 0.0, -0.23 * scale),
        0.105 * scale, 0.46 * scale, materials["lower"], hip,
    )
    knee = _empty("Knee", hip, (0.0, 0.0, -0.46 * scale))
    _cylinder(
        "Shin", (0.0, 0.015, -0.20 * scale),
        0.085 * scale, 0.40 * scale, materials["lower"], knee,
    )
    foot = _cube(
        "Foot", (0.0, -0.07, -0.45 * scale),
        (0.10 * scale, 0.20 * scale, 0.07 * scale),
        materials["shoe"], knee,
    )
    return {"hip": hip, "knee": knee, "foot": foot}


def _build_silhouette(entity: dict, height_scale: float) -> dict:
    material = create_material(
        f"Weak_{entity['id']}", _identity_accent(str(entity["id"])), alpha=0.78,
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
    shadow_material = create_contact_shadow(root, height_scale, str(entity["id"]))
    return {
        "root": root,
        "arms": [],
        "legs": [],
        "wheels": [],
        "steering_wheels": [],
        "wheel_radius": None,
        "materials": [material, shadow_material],
        "visual_height_meters": 1.92 * height_scale,
    }


def _identity_accent(entity_id: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(entity_id.encode("utf-8")).digest()
    return IDENTITY_ACCENT_PALETTE[digest[0] % len(IDENTITY_ACCENT_PALETTE)]


def _readable_color(
    evidence_color: list[float],
    accent: tuple[float, float, float],
) -> tuple[float, float, float] | list[float]:
    if len(evidence_color) != 3:
        return accent
    chroma = max(evidence_color) - min(evidence_color)
    if chroma >= LOW_CHROMA_THRESHOLD:
        return evidence_color
    return tuple(
        round(
            float(channel) * (1.0 - ACCENT_BLEND_STRENGTH)
            + accent[index] * ACCENT_BLEND_STRENGTH,
            4,
        )
        for index, channel in enumerate(evidence_color)
    )


def create_contact_shadow(
    root: bpy.types.Object,
    scale: float,
    entity_id: str,
) -> bpy.types.Material:
    material = create_material(
        f"ContactShadow_{entity_id}",
        (0.008, 0.012, 0.016),
        alpha=CONTACT_SHADOW_ALPHA,
        roughness=1.0,
    )
    bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1.0, fill_type="NGON")
    shadow = bpy.context.object
    shadow.name = f"ContactShadow_{entity_id}"
    shadow.location = (0.0, 0.0, 0.012)
    shadow.scale = (0.34 * scale, 0.16 * scale, 1.0)
    shadow.data.materials.append(material)
    shadow.parent = root
    return material


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


def _soften_mesh(part: bpy.types.Object, width: float) -> None:
    bevel = part.modifiers.new(name="EvidenceSoftEdges", type="BEVEL")
    bevel.width = width
    bevel.segments = 2

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
