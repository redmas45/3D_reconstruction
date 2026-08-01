import math

import bpy

from materials import confidence_color, create_material


MINIMUM_READABLE_ALPHA = 0.90
LOW_CHROMA_THRESHOLD = 0.16
CONFIDENCE_COLOR_BLEND_STRENGTH = 0.64
MINIMUM_SILHOUETTE_WIDTH_METERS = 0.90
MAXIMUM_SILHOUETTE_WIDTH_METERS = 3.20
SILHOUETTE_HEIGHT_METERS = 1.0
BODY_HEIGHT_METERS = 0.56
ROOF_HEIGHT_METERS = 0.34
WHEEL_RADIUS_METERS = 0.16
CONTACT_SHADOW_ALPHA = 0.44
TRUCK_KINDS = {"truck"}
BUS_KINDS = {"bus"}


def build_vehicle(entity: dict) -> dict:
    alpha = 1.0 if entity["fidelity_tier"] == "supported" else MINIMUM_READABLE_ALPHA
    color = entity["appearance"].get(
        "vehicle_color", confidence_color(entity["confidence"]),
    )
    color = _readable_vehicle_color(color, float(entity["confidence"]))
    body_material = create_material(
        f"Vehicle_{entity['id']}", color, alpha=alpha, metallic=0.08, roughness=0.58,
    )
    wheel_material = create_material(
        f"Wheel_{entity['id']}", (0.025, 0.028, 0.032), roughness=0.84,
    )
    wheel_detail_material = create_material(
        f"WheelDetail_{entity['id']}", (0.22, 0.66, 0.70), roughness=0.62,
    )
    window_material = create_material(
        f"Window_{entity['id']}", (0.035, 0.12, 0.16), roughness=0.40,
    )
    root = bpy.data.objects.new(f"Vehicle_{entity['id']}", None)
    bpy.context.collection.objects.link(root)
    width = _silhouette_width(entity)
    _build_vehicle_profile(
        root,
        width,
        str(entity.get("kind", "")),
        body_material,
        window_material,
    )
    wheels = [
        _wheel_disc(
            root,
            -width * 0.31,
            wheel_material,
            wheel_detail_material,
        ),
        _wheel_disc(
            root,
            width * 0.31,
            wheel_material,
            wheel_detail_material,
        ),
    ]
    shadow_material = _contact_shadow(root, width, str(entity["id"]))
    return {
        "root": root,
        "arms": [],
        "legs": [],
        "wheels": wheels,
        "steering_wheels": [],
        "wheel_radius": WHEEL_RADIUS_METERS,
        "wheel_spin_axis": 1,
        "materials": [
            body_material,
            wheel_material,
            wheel_detail_material,
            window_material,
            shadow_material,
        ],
        "visual_height_meters": SILHOUETTE_HEIGHT_METERS,
        "lock_facing_camera": True,
    }


def _silhouette_width(entity: dict) -> float:
    bbox = entity.get("visual_anchor", {}).get("bbox", [])
    if len(bbox) != 4:
        return 1.8
    pixel_width = max(1.0, float(bbox[2]) - float(bbox[0]))
    pixel_height = max(1.0, float(bbox[3]) - float(bbox[1]))
    requested_width = SILHOUETTE_HEIGHT_METERS * pixel_width / pixel_height
    return max(
        MINIMUM_SILHOUETTE_WIDTH_METERS,
        min(MAXIMUM_SILHOUETTE_WIDTH_METERS, requested_width),
    )


def _build_vehicle_profile(
    root: bpy.types.Object,
    width: float,
    kind: str,
    body_material: bpy.types.Material,
    window_material: bpy.types.Material,
) -> None:
    if kind in TRUCK_KINDS:
        _build_truck(root, width, body_material, window_material)
        return
    if kind in BUS_KINDS:
        _build_bus(root, width, body_material, window_material)
        return
    _build_car(root, width, body_material, window_material)


def _build_truck(
    root: bpy.types.Object,
    width: float,
    body_material: bpy.types.Material,
    window_material: bpy.types.Material,
) -> None:
    _box(
        "TruckCargo",
        (-0.13 * width, 0.0, 0.62),
        (0.34 * width, 0.13, 0.38),
        body_material,
        root,
    )
    _box(
        "TruckCab",
        (0.30 * width, 0.0, 0.49),
        (0.20 * width, 0.14, 0.25),
        body_material,
        root,
    )
    _box(
        "TruckChassis",
        (0.0, 0.0, 0.27),
        (0.48 * width, 0.13, 0.08),
        body_material,
        root,
    )
    _box(
        "TruckWindow",
        (0.32 * width, -0.145, 0.58),
        (0.075 * width, 0.015, 0.085),
        window_material,
        root,
    )


def _build_bus(
    root: bpy.types.Object,
    width: float,
    body_material: bpy.types.Material,
    window_material: bpy.types.Material,
) -> None:
    _box(
        "BusBody",
        (0.0, 0.0, 0.55),
        (0.48 * width, 0.14, 0.38),
        body_material,
        root,
    )
    _box(
        "BusWindows",
        (0.0, -0.145, 0.67),
        (0.40 * width, 0.015, 0.12),
        window_material,
        root,
    )


def _build_car(
    root: bpy.types.Object,
    width: float,
    body_material: bpy.types.Material,
    window_material: bpy.types.Material,
) -> None:
    _vehicle_body(root, width, body_material)
    _vehicle_roof(root, width, body_material)
    _box(
        "CarWindow",
        (0.10 * width, -0.145, 0.57),
        (0.18 * width, 0.015, 0.09),
        window_material,
        root,
    )


def _vehicle_body(
    root: bpy.types.Object,
    width: float,
    material: bpy.types.Material,
) -> None:
    _box(
        "VehicleEvidenceBody",
        (0.0, 0.0, BODY_HEIGHT_METERS * 0.5 + WHEEL_RADIUS_METERS),
        (width * 0.5, 0.12, BODY_HEIGHT_METERS * 0.5),
        material,
        root,
    )


def _box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    root: bpy.types.Object,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    part = bpy.context.object
    part.name = name
    part.location = location
    part.scale = scale
    part.data.materials.append(material)
    part.parent = root


def _vehicle_roof(
    root: bpy.types.Object,
    width: float,
    material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    roof = bpy.context.object
    roof.name = "VehicleEvidenceRoof"
    roof.location = (0.18 * width, 0.0, 0.56)
    roof.scale = (width * 0.29, 0.13, ROOF_HEIGHT_METERS * 0.5)
    roof.data.materials.append(material)
    roof.parent = root


def _readable_vehicle_color(
    evidence_color: list[float],
    confidence: float,
) -> tuple[float, float, float] | list[float]:
    if len(evidence_color) != 3:
        return confidence_color(confidence)
    if max(evidence_color) - min(evidence_color) >= LOW_CHROMA_THRESHOLD:
        return evidence_color
    accent = confidence_color(confidence)
    return tuple(
        round(
            float(channel) * (1.0 - CONFIDENCE_COLOR_BLEND_STRENGTH)
            + accent[index] * CONFIDENCE_COLOR_BLEND_STRENGTH,
            4,
        )
        for index, channel in enumerate(evidence_color)
    )


def _wheel_disc(
    root: bpy.types.Object,
    horizontal_position: float,
    material: bpy.types.Material,
    detail_material: bpy.types.Material,
) -> bpy.types.Object:
    wheel_control = bpy.data.objects.new("VehicleWheelControl", None)
    bpy.context.collection.objects.link(wheel_control)
    wheel_control.location = (
        horizontal_position,
        -0.14,
        WHEEL_RADIUS_METERS,
    )
    wheel_control.parent = root
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20,
        radius=WHEEL_RADIUS_METERS,
        depth=0.10,
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    wheel = bpy.context.object
    wheel.name = "VehicleEvidenceWheel"
    wheel.location = (0.0, 0.0, 0.0)
    wheel.data.materials.append(material)
    wheel.parent = wheel_control
    _wheel_spoke(wheel_control, detail_material)
    return wheel_control


def _wheel_spoke(
    wheel_control: bpy.types.Object,
    material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    spoke = bpy.context.object
    spoke.name = "VehicleWheelSpoke"
    spoke.location = (0.0, -0.06, 0.0)
    spoke.scale = (WHEEL_RADIUS_METERS * 0.74, 0.018, 0.018)
    spoke.data.materials.append(material)
    spoke.parent = wheel_control


def _contact_shadow(
    root: bpy.types.Object,
    width: float,
    entity_id: str,
) -> bpy.types.Material:
    material = create_material(
        f"VehicleContactShadow_{entity_id}",
        (0.008, 0.012, 0.016),
        alpha=CONTACT_SHADOW_ALPHA,
        roughness=1.0,
    )
    bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1.0, fill_type="NGON")
    shadow = bpy.context.object
    shadow.name = f"VehicleContactShadow_{entity_id}"
    shadow.location = (0.0, 0.0, 0.012)
    shadow.scale = (width * 0.46, 0.22, 1.0)
    shadow.data.materials.append(material)
    shadow.parent = root
    return material
