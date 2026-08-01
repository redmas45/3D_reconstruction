import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from detect import RELEVANT_COCO_CLASSES
from domain.actor_proxies import (
    CATALOG,
    DEFAULT_PROXY,
    PROXY_HUMANOID,
    SUPPORTED_PROXIES,
    bounding_half_extents,
    catalog_report,
    is_articulated,
    proxy_for,
)
from domain.humanoid_rig import (
    BONE_NAMES,
    SKELETON,
    bone_by_name,
    skeleton_for_height,
    validate_skeleton,
)


class TestCatalogCoverage:
    def test_every_class_the_detector_reports_can_be_drawn(self):
        """The guard against drift: widening the detector must not silently fall back."""
        missing = sorted(set(RELEVANT_COCO_CLASSES.values()) - set(CATALOG))
        assert missing == [], f"classes the detector reports but the renderer cannot draw: {missing}"

    def test_the_catalog_does_not_carry_classes_the_detector_never_reports(self):
        extra = sorted(set(CATALOG) - set(RELEVANT_COCO_CLASSES.values()))
        assert extra == [], f"catalog entries nothing will ever request: {extra}"

    def test_every_entry_names_a_proxy_that_exists(self):
        assert all(spec.proxy in SUPPORTED_PROXIES for spec in CATALOG.values())

    def test_every_entry_has_positive_dimensions(self):
        for name, spec in CATALOG.items():
            assert min(spec.length, spec.width, spec.height) > 0.0, name

    def test_the_report_lists_the_whole_catalog(self):
        assert len(catalog_report()) == len(CATALOG)


class TestLookup:
    def test_a_known_class_resolves(self):
        assert proxy_for("car").proxy == "vehicle"

    def test_lookup_ignores_case_and_padding(self):
        assert proxy_for("  Car ") is proxy_for("car")

    def test_an_unknown_class_falls_back_rather_than_raising(self):
        assert proxy_for("unicorn") is DEFAULT_PROXY

    def test_only_people_are_articulated(self):
        articulated = [name for name in CATALOG if is_articulated(name)]
        assert articulated == ["person"]
        assert proxy_for("person").proxy == PROXY_HUMANOID


class TestBounds:
    def test_a_carried_object_is_bounded_up_to_where_it_is_carried(self):
        """A phone at chest height needs a box reaching chest height, not 15cm."""
        _, _, top = bounding_half_extents("cell phone")
        assert top > 1.2

    def test_a_grounded_object_is_bounded_by_its_own_height(self):
        _, _, top = bounding_half_extents("suitcase")
        assert top == pytest.approx(0.68)

    def test_a_bus_is_bounded_larger_than_a_person(self):
        assert bounding_half_extents("bus")[0] > bounding_half_extents("person")[0]

    def test_an_unknown_class_still_yields_usable_bounds(self):
        half_length, half_width, top = bounding_half_extents("unicorn")
        assert min(half_length, half_width, top) > 0.0


class TestSkeleton:
    def test_the_skeleton_is_structurally_valid(self):
        validate_skeleton()

    def test_both_sides_are_present_and_mirrored(self):
        left = {name for name in BONE_NAMES if name.endswith(".L")}
        right = {name for name in BONE_NAMES if name.endswith(".R")}
        assert {name[:-2] for name in left} == {name[:-2] for name in right}

    def test_mirrored_bones_have_opposite_x(self):
        bones = bone_by_name()
        assert bones["thigh.L"].head[0] == pytest.approx(-bones["thigh.R"].head[0])

    def test_mirrored_bones_keep_their_parents_on_their_own_side(self):
        assert bone_by_name()["forearm.R"].parent == "upper_arm.R"

    def test_the_figure_stands_on_the_ground(self):
        """Feet at ground level. A figure floating or sunk is immediately obvious."""
        lowest = min(min(bone.head[2], bone.tail[2]) for bone in SKELETON)
        assert 0.0 <= lowest <= 0.06

    def test_the_figure_is_the_reference_height(self):
        assert max(max(bone.head[2], bone.tail[2]) for bone in SKELETON) == pytest.approx(1.75)

    def test_scaling_preserves_proportions(self):
        tall = skeleton_for_height(2.10)
        assert max(max(b.head[2], b.tail[2]) for b in tall) == pytest.approx(2.10)
        validate_skeleton(tall)

    def test_arms_hang_clear_of_the_torso(self):
        """An arm inside the chest reads as a figure with no arms."""
        bones = bone_by_name()
        assert bones["upper_arm.L"].head[0] > bones["chest"].head[0] + 0.18

    def test_every_bone_the_gait_poses_exists_on_the_skeleton(self):
        from domain.gait import walk_pose

        posed = set(walk_pose(1.0, 1.4).rotations) | set(walk_pose(0.0, 0.0).rotations)
        assert posed <= set(BONE_NAMES), sorted(posed - set(BONE_NAMES))
