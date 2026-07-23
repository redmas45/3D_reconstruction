import bpy

from materials import confidence_color, create_material


VEHICLE_DIMENSIONS = {
    "car": (1.75, 3.8, 1.25),
    "truck": (2.2, 5.6, 1.8),
    "bus": (2.4, 8.0, 2.8),
    "motorcycle": (0.7, 2.0, 1.1),
    "bicycle": (0.55, 1.8, 1.1),
}
WHEEL_RADIUS_METERS = 0.34


def build_vehicle(entity: dict) -> dict:
    width, length, height = VEHICLE_DIMENSIONS.get(entity["kind"], VEHICLE_DIMENSIONS["car"])
    alpha = 1.0 if entity["fidelity_tier"] == "supported" else 0.62
    color = entity["appearance"].get("vehicle_color", confidence_color(entity["confidence"]))
    body_material = create_material(f"Vehicle_{entity['id']}", color, alpha=alpha, metallic=0.22, roughness=0.34)
    wheel_material = create_material(f"Wheel_{entity['id']}", (0.025, 0.028, 0.032), roughness=0.8)
    root = bpy.data.objects.new(f"Vehicle_{entity['id']}", None)
    bpy.context.collection.objects.link(root)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, height * 0.55))
    body = bpy.context.object
    body.scale = (width * 0.5, length * 0.5, height * 0.5)
    body.data.materials.append(body_material)
    body.parent = root
    wheels, steering_wheels = _build_wheels(width, length, wheel_material, root)
    return {
        "root": root,
        "arms": [],
        "legs": [],
        "wheels": wheels,
        "steering_wheels": steering_wheels,
        "wheel_radius": WHEEL_RADIUS_METERS,
        "materials": [body_material, wheel_material],
    }


def _build_wheels(
    width: float,
    length: float,
    material: bpy.types.Material,
    root: bpy.types.Object,
) -> tuple[list[bpy.types.Object], list[bpy.types.Object]]:
    wheels: list[bpy.types.Object] = []
    steering_wheels: list[bpy.types.Object] = []
    for side in (-1.0, 1.0):
        for longitudinal in (-1.0, 1.0):
            location = (
                side * width * 0.52,
                longitudinal * length * 0.32,
                WHEEL_RADIUS_METERS,
            )
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=16,
                radius=WHEEL_RADIUS_METERS,
                depth=0.18,
                location=location,
                rotation=(0.0, 1.5708, 0.0),
            )
            wheel = bpy.context.object
            wheel.name = f"{'Front' if longitudinal > 0.0 else 'Rear'}Wheel"
            wheel.data.materials.append(material)
            wheel.parent = root
            wheels.append(wheel)
            if longitudinal > 0.0:
                steering_wheels.append(wheel)
    return wheels, steering_wheels
