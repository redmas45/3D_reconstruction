import bpy

from materials import create_material


GROUND_SIZE_METERS = 34.0
GRID_HALF_EXTENT = 16
GRID_STEP_METERS = 2
STREET_PROXY_PROFILE = "street"


def build_environment(plan: dict) -> None:
    environment = plan["environment"]
    ground_material = create_material("Ground", environment["ground_color"], roughness=0.78)
    grid_material = create_material("EvidenceGrid", environment["grid_color"], alpha=0.34, metallic=0.15)
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE_METERS, location=(0.0, 10.0, 0.0))
    ground = bpy.context.object
    ground.name = "Forensic_Ground"
    ground.data.materials.append(ground_material)
    for coordinate in range(-GRID_HALF_EXTENT, GRID_HALF_EXTENT + 1, GRID_STEP_METERS):
        _grid_line((-GRID_HALF_EXTENT, coordinate + 10.0, 0.012), (GRID_HALF_EXTENT, coordinate + 10.0, 0.012), grid_material)
        _grid_line((coordinate, -GRID_HALF_EXTENT + 10.0, 0.012), (coordinate, GRID_HALF_EXTENT + 10.0, 0.012), grid_material)
    if environment.get("proxy_profile") == STREET_PROXY_PROFILE:
        _build_street_proxies()


def build_path_trail(entity: dict) -> None:
    waypoints = [item["world"] for item in entity["path_prediction"]["waypoints"]]
    curve_data = bpy.data.curves.new(f"Path_{entity['id']}", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.025 if entity["fidelity_tier"] == "supported" else 0.045
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(waypoints) - 1)
    for point, world in zip(spline.bezier_points, waypoints):
        point.co = (world[0], world[1], 0.035)
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    curve_object = bpy.data.objects.new(f"Path_{entity['id']}", curve_data)
    bpy.context.collection.objects.link(curve_object)
    color = (0.08, 0.82, 0.58) if entity["confidence"] >= 0.75 else (0.96, 0.61, 0.08)
    curve_object.data.materials.append(create_material(f"PathMaterial_{entity['id']}", color, alpha=0.62))


def _grid_line(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    material: bpy.types.Material,
) -> None:
    curve_data = bpy.data.curves.new("GridLine", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.008
    spline = curve_data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (*start, 1.0)
    spline.points[1].co = (*end, 1.0)
    line = bpy.data.objects.new("GridLine", curve_data)
    bpy.context.collection.objects.link(line)
    line.data.materials.append(material)


def _build_street_proxies() -> None:
    facade_material = create_material("ProxyFacade", (0.025, 0.045, 0.065), roughness=0.82)
    accent_material = create_material("ProxyAccent", (0.03, 0.32, 0.36), alpha=0.48, metallic=0.12)
    storefront_material = create_material("ProxyStorefront", (0.035, 0.11, 0.15), roughness=0.38, metallic=0.18)
    for side in (-1.0, 1.0):
        _proxy_block((side * 8.0, 11.0, 4.0), (2.2, 25.0, 8.0), facade_material)
        _proxy_block((side * 6.75, 11.0, 0.20), (0.18, 22.0, 0.40), accent_material)
        for street_depth in (5.0, 12.0, 19.0):
            _proxy_block((side * 6.95, street_depth, 2.0), (0.65, 4.8, 3.8), storefront_material)
            _proxy_block((side * 6.58, street_depth, 3.15), (0.10, 3.4, 0.18), accent_material)
    for side in (-1.0, 1.0):
        for street_depth in (4.0, 14.0):
            _street_lamp(side * 5.7, street_depth, accent_material)


def _proxy_block(
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    proxy = bpy.context.object
    proxy.name = "Scene_Context_Proxy"
    proxy.dimensions = dimensions
    proxy.data.materials.append(material)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def _street_lamp(
    horizontal_position: float, street_depth: float, material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=12, radius=0.055, depth=3.2, location=(horizontal_position, street_depth, 1.6)
    )
    pole = bpy.context.object
    pole.name = "Street_Context_Lamp"
    pole.data.materials.append(material)
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2, radius=0.16, location=(horizontal_position, street_depth, 3.25)
    )
    lamp = bpy.context.object
    lamp.name = "Street_Context_Light"
    lamp.data.materials.append(material)
