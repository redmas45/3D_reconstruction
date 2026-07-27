"""The prebuilt actor asset library: what is in it, and whether it is still valid.

Geometry is currently generated inside every Blender process from `actor_proxies`. That
is cheap today because the models are simple — and that is exactly the constraint worth
removing. Building the models once, into a library Blender links rather than generates,
means model quality stops being paid for on every run. A far more detailed figure costs
the same at render time as a crude one.

Three properties make the library safe to rely on:

  * **It is optional.** When the library is absent, or was built from a different
    catalog, the renderer generates geometry procedurally exactly as before. The repo
    works from a clean checkout with no build step.
  * **It is content-addressed.** The manifest carries a digest of the catalog it was
    built from. A changed proxy, a new class, or a changed skeleton produces a different
    digest, so a stale library is detected rather than silently used.
  * **It feeds the render cache key.** The digest travels in the job manifest, so
    rebuilding the library invalidates rendered layers that were produced with the old
    models instead of mixing the two in one video.

Adding a class the detector newly reports is a row in `actor_proxies.CATALOG` and a
rebuild — no code change here or in Blender.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from domain.actor_proxies import CATALOG, PROXY_HUMANOID, ProxySpec
from domain.humanoid_rig import SKELETON


LIBRARY_SCHEMA_VERSION = 1
LIBRARY_BLEND_NAME = "library.blend"
LIBRARY_MANIFEST_NAME = "manifest.json"
# Objects in the library are named by class, with spaces made safe for a datablock name.
OBJECT_NAME_PREFIX = "FOR3D_Asset"


class ActorLibraryError(RuntimeError):
    """The asset library is present but unusable."""


def asset_object_name(class_name: str) -> str:
    return f"{OBJECT_NAME_PREFIX}_{str(class_name).strip().lower().replace(' ', '_')}"


def asset_object_names(class_name: str, spec: ProxySpec) -> list[str]:
    """Every object that must be appended for this asset to render.

    An articulated actor is two objects — the rig that carries the pose and the skinned
    body parented to it. Appending only the rig produces an actor with no geometry that
    renders as an empty frame, so the whole set is declared here rather than inferred
    from a name suffix at load time.
    """
    primary = asset_object_name(class_name)
    if spec.proxy == PROXY_HUMANOID:
        return [primary, f"{primary}_body"]
    return [primary]


def _proxy_fingerprint(name: str, spec: ProxySpec) -> dict:
    return {
        "class_name": name,
        "proxy": spec.proxy,
        "dimensions": [spec.length, spec.width, spec.height],
        "ground_offset": spec.ground_offset_meters,
        "body_height_ratio": spec.body_height_ratio,
        "cabin_length_ratio": spec.cabin_length_ratio,
        "cabin_width_ratio": spec.cabin_width_ratio,
        "object_name": asset_object_name(name),
        "object_names": asset_object_names(name, spec),
    }


def catalog_digest() -> str:
    """Identifies the geometry the catalog and skeleton together describe.

    The skeleton is included because it drives the articulated mesh: moving a shoulder
    changes the model without changing any proxy dimension, and a library built before
    that change must not be reused after it.
    """
    payload = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "proxies": [_proxy_fingerprint(name, spec) for name, spec in sorted(CATALOG.items())],
        "skeleton": [
            [bone.name, bone.parent, list(bone.head), list(bone.tail)] for bone in SKELETON
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def build_manifest() -> dict:
    """Everything a build must produce, and everything a loader must check."""
    return {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "catalog_digest": catalog_digest(),
        "assets": [_proxy_fingerprint(name, spec) for name, spec in sorted(CATALOG.items())],
    }


@dataclass(frozen=True)
class ActorLibrary:
    """A validated library on disk."""

    blend_path: Path
    manifest: dict

    @property
    def digest(self) -> str:
        return str(self.manifest["catalog_digest"])

    def object_name_for(self, class_name: str) -> str | None:
        """The object a class's actors are copied from, or None when absent."""
        wanted = asset_object_name(class_name)
        for asset in self.manifest.get("assets", []):
            if asset.get("object_name") == wanted:
                return wanted
        return None

    def objects_to_append(self, class_names) -> list[str]:
        """Every object that must be appended to render these classes."""
        wanted = {asset_object_name(name) for name in class_names}
        names: list[str] = []
        for asset in self.manifest.get("assets", []):
            if asset.get("object_name") in wanted:
                names.extend(asset.get("object_names", [asset["object_name"]]))
        return sorted(set(names))

    def covers(self, class_names) -> bool:
        return all(self.object_name_for(name) is not None for name in class_names)


def write_manifest(directory: Path, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LIBRARY_MANIFEST_NAME
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def load_library(directory: Path) -> ActorLibrary | None:
    """Return the library only when it is present, parseable, and current.

    Every failure returns None rather than raising: a missing or stale library is a
    normal state that falls back to procedural generation, not an error. The one thing
    it must never do is return a library built from different geometry.
    """
    blend_path = Path(directory) / LIBRARY_BLEND_NAME
    manifest_path = Path(directory) / LIBRARY_MANIFEST_NAME
    if not blend_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("schema_version") != LIBRARY_SCHEMA_VERSION:
        return None
    if manifest.get("catalog_digest") != catalog_digest():
        return None
    return ActorLibrary(blend_path=blend_path, manifest=manifest)


def library_state(directory: Path) -> dict:
    """A description of the library for the run report, whatever state it is in."""
    library = load_library(directory)
    if library is not None:
        return {
            "available": True,
            "catalog_digest": library.digest,
            "asset_count": len(library.manifest.get("assets", [])),
            "path": str(library.blend_path),
        }
    blend_path = Path(directory) / LIBRARY_BLEND_NAME
    return {
        "available": False,
        "catalog_digest": catalog_digest(),
        "reason": "stale" if blend_path.is_file() else "not built",
        "path": str(blend_path),
    }
