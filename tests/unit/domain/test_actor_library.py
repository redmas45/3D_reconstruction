import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.actor_library import (
    LIBRARY_BLEND_NAME,
    LIBRARY_MANIFEST_NAME,
    LIBRARY_SCHEMA_VERSION,
    asset_object_name,
    build_manifest,
    catalog_digest,
    library_state,
    load_library,
    write_manifest,
)
from domain.actor_proxies import CATALOG


def _build(directory: Path, manifest=None) -> Path:
    (directory / LIBRARY_BLEND_NAME).write_bytes(b"BLENDER-v450" + b"\x00" * 64)
    write_manifest(directory, manifest or build_manifest())
    return directory


class TestManifest:
    def test_the_manifest_covers_every_catalog_class(self):
        assets = {asset["class_name"] for asset in build_manifest()["assets"]}
        assert assets == set(CATALOG)

    def test_object_names_are_datablock_safe(self):
        assert asset_object_name("cell phone") == "FOR3D_Asset_cell_phone"

    def test_object_names_ignore_case_and_padding(self):
        assert asset_object_name(" Cell Phone ") == asset_object_name("cell phone")

    def test_every_asset_carries_the_geometry_it_needs(self):
        for asset in build_manifest()["assets"]:
            assert len(asset["dimensions"]) == 3
            assert asset["proxy"]
            assert asset["object_name"]


class TestDigest:
    def test_the_digest_is_stable_across_calls(self):
        assert catalog_digest() == catalog_digest()

    def test_the_digest_changes_when_a_proxy_changes(self, monkeypatch):
        before = catalog_digest()
        widened = dict(CATALOG)
        widened["car"] = CATALOG["car"].__class__(
            **{**CATALOG["car"].__dict__, "length": 9.9},
        )
        monkeypatch.setattr("domain.actor_library.CATALOG", widened)
        assert catalog_digest() != before

    def test_the_digest_changes_when_the_skeleton_changes(self, monkeypatch):
        """A moved shoulder changes the model without changing any proxy dimension."""
        from domain.humanoid_rig import SKELETON

        before = catalog_digest()
        moved = tuple(bone.scaled(1.05) for bone in SKELETON)
        monkeypatch.setattr("domain.actor_library.SKELETON", moved)
        assert catalog_digest() != before


class TestLoading:
    def test_a_freshly_built_library_loads(self, tmp_path):
        library = load_library(_build(tmp_path))
        assert library is not None
        assert library.digest == catalog_digest()

    def test_an_absent_library_is_not_an_error(self, tmp_path):
        assert load_library(tmp_path) is None

    def test_a_library_without_its_blend_is_ignored(self, tmp_path):
        write_manifest(tmp_path, build_manifest())
        assert load_library(tmp_path) is None

    def test_a_stale_library_is_refused(self, tmp_path):
        """The one thing it must never do: use geometry it was not built from."""
        manifest = build_manifest()
        manifest["catalog_digest"] = "0000000000000000"
        assert load_library(_build(tmp_path, manifest)) is None

    def test_a_library_from_a_future_schema_is_refused(self, tmp_path):
        manifest = build_manifest()
        manifest["schema_version"] = LIBRARY_SCHEMA_VERSION + 1
        assert load_library(_build(tmp_path, manifest)) is None

    def test_a_corrupt_manifest_is_refused_rather_than_raising(self, tmp_path):
        (tmp_path / LIBRARY_BLEND_NAME).write_bytes(b"x" * 64)
        (tmp_path / LIBRARY_MANIFEST_NAME).write_text("{not json", encoding="utf-8")
        assert load_library(tmp_path) is None

    def test_the_library_reports_which_classes_it_covers(self, tmp_path):
        library = load_library(_build(tmp_path))
        assert library.covers(["person", "car"])
        assert not library.covers(["unicorn"])

    def test_no_partial_manifest_is_left_behind(self, tmp_path):
        write_manifest(tmp_path, build_manifest())
        assert not any(".writing" in path.name for path in tmp_path.iterdir())


class TestState:
    def test_an_unbuilt_library_says_so(self, tmp_path):
        state = library_state(tmp_path)
        assert state["available"] is False
        assert state["reason"] == "not built"

    def test_a_stale_library_is_distinguished_from_a_missing_one(self, tmp_path):
        manifest = build_manifest()
        manifest["catalog_digest"] = "0000000000000000"
        state = library_state(_build(tmp_path, manifest))
        assert state["available"] is False
        assert state["reason"] == "stale"

    def test_a_current_library_reports_its_contents(self, tmp_path):
        state = library_state(_build(tmp_path))
        assert state["available"] is True
        assert state["asset_count"] == len(CATALOG)


class TestCheckedInLibrary:
    """The library committed to the repository, if one has been built."""

    LIBRARY_DIRECTORY = Path(__file__).resolve().parents[3] / "assets" / "actors"

    def test_a_built_library_in_the_repository_is_current(self):
        manifest_path = self.LIBRARY_DIRECTORY / LIBRARY_MANIFEST_NAME
        if not manifest_path.is_file():
            return  # Not built in this checkout, which is a supported state.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["catalog_digest"] == catalog_digest(), (
            "assets/actors is stale — rerun scripts/build_actor_library.py"
        )
