"""What to draw for each class YOLO can detect, and how big it is.

One catalog, consulted by everything that needs to know an entity's physical extent:
the render region projects it to find the crop, and Blender builds geometry from it.
Two tables would mean an actor whose crop was computed for a car and whose mesh was
built for a bicycle.

**Universal by construction.** The catalog covers exactly the classes the detector
reports (`detect.RELEVANT_COCO_CLASSES`), enforced by a test so the two cannot drift.
Anything unrecognised falls back to a plausible grounded box rather than failing, so
widening the detector's class list degrades gracefully instead of crashing. Adding a
class properly is a row here, not a code path.

Only *moving* entities reach this catalog. Everything static — street furniture,
parked vehicles, the scene itself — is already present in the photographic plate
recovered from the visible footage, and drawing it again would double it.

Dimensions are the object's real-world extent in metres, in its own frame before it is
rotated to face its direction of travel:

  * `length` spans X, the object's left-to-right axis
  * `width` spans Y, its front-to-back depth
  * `height` spans Z, upward from its origin

`ground_offset_meters` is how far the origin sits above the ground. Zero for anything
resting on it; roughly hip or chest height for things a person carries, because a
handbag floating at ankle level reads as a bug even when its position is right.
"""

from dataclasses import dataclass


PROXY_HUMANOID = "humanoid"
PROXY_VEHICLE = "vehicle"
PROXY_BOX = "box"
PROXY_CYLINDER = "cylinder"

SUPPORTED_PROXIES = frozenset({PROXY_HUMANOID, PROXY_VEHICLE, PROXY_BOX, PROXY_CYLINDER})


@dataclass(frozen=True)
class ProxySpec:
    proxy: str
    length: float
    width: float
    height: float
    ground_offset_meters: float = 0.0
    # Only wheeled proxies have a cabin; the ratio is what makes a bus read as a bus and
    # a bicycle as a bicycle rather than both being boxes of different sizes.
    body_height_ratio: float = 0.62
    cabin_length_ratio: float = 0.55
    cabin_width_ratio: float = 0.92

    @property
    def top_meters(self) -> float:
        return self.ground_offset_meters + self.height

    @property
    def half_extents(self) -> tuple[float, float]:
        return self.length / 2.0, self.width / 2.0


DEFAULT_PROXY = ProxySpec(PROXY_BOX, 0.50, 0.50, 0.60)

CATALOG: dict[str, ProxySpec] = {
    "person": ProxySpec(PROXY_HUMANOID, 0.50, 0.34, 1.75),

    # Wheeled. Cabin ratios are tuned so the silhouette is recognisable at the size these
    # occupy in a frame, which is the only place it matters.
    "bicycle": ProxySpec(PROXY_VEHICLE, 1.75, 0.50, 1.10, body_height_ratio=0.55,
                         cabin_length_ratio=0.30, cabin_width_ratio=0.35),
    "motorcycle": ProxySpec(PROXY_VEHICLE, 2.10, 0.80, 1.30, body_height_ratio=0.58,
                            cabin_length_ratio=0.40, cabin_width_ratio=0.55),
    "car": ProxySpec(PROXY_VEHICLE, 4.30, 1.80, 1.45, body_height_ratio=0.60,
                     cabin_length_ratio=0.52, cabin_width_ratio=0.90),
    "bus": ProxySpec(PROXY_VEHICLE, 10.50, 2.50, 3.20, body_height_ratio=0.88,
                     cabin_length_ratio=0.96, cabin_width_ratio=0.98),
    "truck": ProxySpec(PROXY_VEHICLE, 7.00, 2.40, 2.90, body_height_ratio=0.72,
                       cabin_length_ratio=0.34, cabin_width_ratio=0.96),

    # Carried and placed objects. Small, but they are evidence and the system is not
    # entitled to decide they are too small to reconstruct.
    "backpack": ProxySpec(PROXY_BOX, 0.32, 0.22, 0.46, ground_offset_meters=0.95),
    "handbag": ProxySpec(PROXY_BOX, 0.35, 0.15, 0.28, ground_offset_meters=0.85),
    "suitcase": ProxySpec(PROXY_BOX, 0.45, 0.25, 0.68),
    "bottle": ProxySpec(PROXY_CYLINDER, 0.08, 0.08, 0.26, ground_offset_meters=0.95),
    "cup": ProxySpec(PROXY_CYLINDER, 0.09, 0.09, 0.12, ground_offset_meters=1.00),
    "knife": ProxySpec(PROXY_BOX, 0.04, 0.02, 0.24, ground_offset_meters=0.95),
    "cell phone": ProxySpec(PROXY_BOX, 0.07, 0.012, 0.15, ground_offset_meters=1.15),
}



def proxy_for(class_name: str) -> ProxySpec:
    """The proxy for a detected class, or a plausible default for anything unknown."""
    return CATALOG.get(str(class_name).strip().lower(), DEFAULT_PROXY)


def is_articulated(class_name: str) -> bool:
    """Whether this class is drawn with a rig and a gait rather than as rigid geometry."""
    return proxy_for(class_name).proxy == PROXY_HUMANOID


def bounding_half_extents(class_name: str) -> tuple[float, float, float]:
    """Half-length, half-width and top height, as the render region needs them.

    Returns the *top* rather than the height so the caller does not have to know about
    `ground_offset_meters` — a carried phone's box must reach up to chest height, not
    only span the 15 cm the phone itself occupies.
    """
    spec = proxy_for(class_name)
    half_length, half_width = spec.half_extents
    return half_length, half_width, spec.top_meters


def catalog_report() -> list[dict]:
    """Every class the renderer can draw, for the run report and the UI."""
    return [
        {
            "class_name": name,
            "proxy": spec.proxy,
            "dimensions_meters": [spec.length, spec.width, spec.height],
            "ground_offset_meters": spec.ground_offset_meters,
        }
        for name, spec in sorted(CATALOG.items())
    ]
