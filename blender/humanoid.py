"""Builds the rigged humanoid actor inside Blender.

One mesh and one armature datablock are built per job and shared by every actor. Each
actor gets its own *object* pair — an armature object carries pose, so it cannot be
shared — but both point at the same data, so a fifth actor costs almost nothing.

The mesh is generated from `humanoid_rig.SKELETON` rather than authored, so the surface
and the bones cannot disagree: every limb is a tapered tube around the bone that drives
it, and the vertex weights are assigned from that same relationship rather than from
Blender's automatic weighting, which needs a scene evaluation and is not deterministic
enough to test against.

Deliberately a clean stylised figure rather than an attempt at a specific person. The
system reconstructs where someone was and how fast they moved; rendering a detailed
likeness would assert identity information the evidence does not contain.
"""

import math

import bpy
from mathutils import Matrix, Vector


HUMANOID_MESH_NAME = "FOR3D_HumanoidMesh"
HUMANOID_ARMATURE_NAME = "FOR3D_HumanoidRig"

# Radius of the tube drawn around each bone, in metres at reference height. Tapered from
# head to tail so limbs narrow toward the extremities the way real ones do.
BONE_RADII = {
    "pelvis": (0.155, 0.145),
    "chest": (0.150, 0.135),
    "neck": (0.058, 0.055),
    "head": (0.098, 0.078),
    "clavicle": (0.055, 0.050),
    "upper_arm": (0.058, 0.046),
    "forearm": (0.045, 0.034),
    "hand": (0.036, 0.026),
    "thigh": (0.088, 0.064),
    "shin": (0.062, 0.042),
    "foot": (0.045, 0.038),
}
DEFAULT_BONE_RADIUS = (0.05, 0.05)

# Sides of each limb tube. Ten reads as round once smooth-shaded and subdivided.
TUBE_SIDES = 10
# Where the support loops sit, as a fraction of bone length in from each end.
SUPPORT_LOOP_FRACTION = 0.12
# How far segments extend past their joints so consecutive limbs interpenetrate.
JOINT_OVERLAP_METERS = 0.035
JOINT_OVERLAP_RATIO = 0.30

SUBDIVISION_LEVELS = 1


def _bone_radius(bone_name: str) -> tuple[float, float]:
    base = bone_name.split(".")[0]
    return BONE_RADII.get(base, DEFAULT_BONE_RADIUS)


def _ring(centre: Vector, direction: Vector, radius: float) -> list[Vector]:
    """A ring of vertices perpendicular to `direction`, centred on `centre`."""
    axis = direction.normalized()
    # Any vector not parallel to the axis works as a seed for the perpendicular basis.
    seed = Vector((0.0, 0.0, 1.0)) if abs(axis.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    side = axis.cross(seed).normalized()
    up = axis.cross(side).normalized()
    return [
        centre + (side * math.cos(angle) + up * math.sin(angle)) * radius
        for angle in (
            2.0 * math.pi * index / TUBE_SIDES for index in range(TUBE_SIDES)
        )
    ]


def _tube(bone, scale: float, extend_tail: bool = True) -> tuple[list[Vector], list[tuple[int, ...]], str]:
    """Vertices and faces for one bone's limb segment.

    Two details stop the figure falling apart, both learned from it doing exactly that.

    **Support loops.** A tube made of only two rings collapses under subdivision — the
    surface pulls toward the average of its neighbours and the ends shrink inward, so
    every limb visibly detached from the next. Rings just inside each end hold the shape.

    **Overlap.** Segments are extended past their joints so consecutive limbs
    interpenetrate. Under subdivision the union reads as one continuous body instead of
    a string of separate capsules with gaps at every joint.
    """
    head = Vector(bone.head) * scale
    tail = Vector(bone.tail) * scale
    direction = tail - head
    length = direction.length
    if length <= 0.0:
        return [], [], bone.name
    head_radius, tail_radius = (value * scale for value in _bone_radius(bone.name))
    overlap = min(JOINT_OVERLAP_RATIO, JOINT_OVERLAP_METERS * scale / length)
    # A bone with no child has nothing to merge into, and extending it just makes the
    # part longer than it should be — which is how the feet ended up as paddles.
    tail_overlap = overlap if extend_tail else 0.0
    positions = (
        -overlap, SUPPORT_LOOP_FRACTION, 1.0 - SUPPORT_LOOP_FRACTION, 1.0 + tail_overlap,
    )
    vertices: list[Vector] = []
    for position in positions:
        clamped = max(0.0, min(1.0, position))
        radius = head_radius + (tail_radius - head_radius) * clamped
        vertices.extend(_ring(head + direction * position, direction, radius))
    faces = []
    for ring in range(len(positions) - 1):
        base = ring * TUBE_SIDES
        faces.extend(
            (base + index, base + (index + 1) % TUBE_SIDES,
             base + TUBE_SIDES + (index + 1) % TUBE_SIDES, base + TUBE_SIDES + index)
            for index in range(TUBE_SIDES)
        )
    last = (len(positions) - 1) * TUBE_SIDES
    faces.append(tuple(range(last, last + TUBE_SIDES)))
    faces.append(tuple(reversed(range(TUBE_SIDES))))
    return vertices, faces, bone.name


def build_humanoid_mesh(skeleton, scale: float = 1.0) -> bpy.types.Mesh:
    """One mesh covering every bone, with a vertex group per bone already weighted."""
    existing = bpy.data.meshes.get(HUMANOID_MESH_NAME)
    if existing is not None:
        return existing
    vertices: list[Vector] = []
    faces: list[tuple[int, ...]] = []
    groups: dict[str, list[int]] = {}
    parents = {bone.parent for bone in skeleton if bone.parent is not None}
    for bone in skeleton:
        tube_vertices, tube_faces, bone_name = _tube(
            bone, scale, extend_tail=bone.name in parents,
        )
        offset = len(vertices)
        vertices.extend(tube_vertices)
        faces.extend(tuple(index + offset for index in face) for face in tube_faces)
        groups.setdefault(bone_name, []).extend(
            range(offset, offset + len(tube_vertices))
        )
    mesh = bpy.data.meshes.new(HUMANOID_MESH_NAME)
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], faces)
    mesh.validate()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.update()
    mesh["for3d_vertex_groups"] = {name: list(indices) for name, indices in groups.items()}
    return mesh


def build_humanoid_armature(skeleton, scale: float = 1.0) -> bpy.types.Armature:
    """The armature datablock. Shared by every actor; pose lives on the objects."""
    existing = bpy.data.armatures.get(HUMANOID_ARMATURE_NAME)
    if existing is not None:
        return existing
    armature = bpy.data.armatures.new(HUMANOID_ARMATURE_NAME)
    rig_object = bpy.data.objects.new("FOR3D_RigBuilder", armature)
    bpy.context.scene.collection.objects.link(rig_object)
    bpy.context.view_layer.objects.active = rig_object
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        created = {}
        for bone in skeleton:
            edit_bone = armature.edit_bones.new(bone.name)
            edit_bone.head = tuple(value * scale for value in bone.head)
            edit_bone.tail = tuple(value * scale for value in bone.tail)
            edit_bone.use_connect = False
            if bone.parent is not None:
                edit_bone.parent = created[bone.parent]
            created[bone.name] = edit_bone
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(rig_object, do_unlink=True)
    return armature


def instance_humanoid(
    name: str, mesh: bpy.types.Mesh, armature: bpy.types.Armature,
    material: bpy.types.Material, collection: bpy.types.Collection,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Create one actor: a rig object plus a skinned mesh object bound to it."""
    rig_object = bpy.data.objects.new(f"{name}_rig", armature)
    collection.objects.link(rig_object)
    body = bpy.data.objects.new(name, mesh)
    collection.objects.link(body)
    for group_name, indices in (mesh.get("for3d_vertex_groups") or {}).items():
        group = body.vertex_groups.new(name=group_name)
        # Weight 1.0 and one group per vertex: every vertex belongs to exactly the bone
        # whose tube it was generated from, so the binding is exact by construction.
        group.add(list(indices), 1.0, "REPLACE")
    if not body.data.materials:
        body.data.materials.append(material)
    modifier = body.modifiers.new(name="FOR3D_Armature", type="ARMATURE")
    modifier.object = rig_object
    body.parent = rig_object
    # `Object.pose` does not exist until the depsgraph has seen the new object, and
    # posing silently does nothing until it does. This one line is the difference
    # between a walking figure and a T-posed one.
    bpy.context.view_layer.update()
    if rig_object.pose is None:
        raise RuntimeError(f"Armature object {rig_object.name} has no pose to drive")
    return rig_object, body


def apply_pose(
    rig_object: bpy.types.Object, rotations: dict, root_offset=(0.0, 0.0, 0.0),
) -> int:
    """Set every named bone's rotation for the current frame.

    Raises rather than skipping when nothing could be posed. A figure that renders in
    its rest pose looks like a deliberate choice, so a silent failure here survives
    every structural check and only shows up as "why isn't it walking".
    """
    if rig_object.pose is None:
        raise RuntimeError(f"{rig_object.name} has no pose; the rig was never evaluated")
    posed = 0
    unknown = []
    for bone_name, rotation in rotations.items():
        pose_bone = rig_object.pose.bones.get(bone_name)
        if pose_bone is None:
            unknown.append(bone_name)
            continue
        pose_bone.rotation_mode = "XYZ"
        pose_bone.rotation_euler = tuple(float(angle) for angle in rotation)
        if bone_name == "pelvis":
            pose_bone.location = _pelvis_offset(pose_bone, root_offset)
        posed += 1
    if rotations and not posed:
        raise RuntimeError(
            f"None of the {len(rotations)} posed bones exist on {rig_object.name}: "
            f"{sorted(unknown)[:6]}"
        )
    return posed


def _pelvis_offset(pose_bone: bpy.types.PoseBone, root_offset) -> Vector:
    """Convert a world-space offset into the pelvis bone's own space.

    Pose-bone location is expressed along the bone's local axes, and the pelvis points
    up, so a raw world Z offset would move the figure sideways instead of down.
    """
    world = Vector(tuple(float(value) for value in root_offset))
    return pose_bone.bone.matrix_local.to_3x3().inverted() @ world


def keyframe_pose(rig_object: bpy.types.Object, frame: float, rotations: dict) -> None:
    for bone_name in rotations:
        pose_bone = rig_object.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame)
        if bone_name == "pelvis":
            pose_bone.keyframe_insert(data_path="location", frame=frame)


def add_subdivision(body: bpy.types.Object) -> None:
    """Round off the tube silhouettes. One level is enough at the size a figure occupies."""
    if SUBDIVISION_LEVELS <= 0:
        return
    modifier = body.modifiers.new(name="FOR3D_Subdivision", type="SUBSURF")
    modifier.levels = SUBDIVISION_LEVELS
    modifier.render_levels = SUBDIVISION_LEVELS


def apply_matrix(instance: bpy.types.Object, location, heading_degrees: float) -> None:
    instance.matrix_basis = (
        Matrix.Translation(Vector(tuple(float(value) for value in location)))
        @ Matrix.Rotation(math.radians(float(heading_degrees)), 4, "Z")
    )
