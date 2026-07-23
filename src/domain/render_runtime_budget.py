"""Predicts a full Blender run from one representative gap."""

import json
from dataclasses import dataclass
from pathlib import Path


MINIMUM_GAP_WEIGHT = 1.0
DETAILED_ENTITY_WEIGHT = 0.10
WEAK_ENTITY_WEIGHT = 0.03


class RenderRuntimeBudgetExceeded(RuntimeError):
    pass


class RepresentativePreviewApprovalRequired(RuntimeError):
    def __init__(self, preview_path: Path, approval_path: Path, signature: str) -> None:
        super().__init__(
            "Representative render is ready for visual approval before remaining gaps start"
        )
        self.preview_path = preview_path
        self.approval_path = approval_path
        self.signature = signature


@dataclass(frozen=True)
class GapRenderCost:
    gap_index: int
    target_frames: int
    detailed_entities: int
    weak_entities: int

    @property
    def weight(self) -> float:
        entity_factor = (
            1.0
            + self.detailed_entities * DETAILED_ENTITY_WEIGHT
            + self.weak_entities * WEAK_ENTITY_WEIGHT
        )
        return max(MINIMUM_GAP_WEIGHT, self.target_frames * entity_factor)


def gap_render_costs(plans: list[dict]) -> list[GapRenderCost]:
    return [_gap_render_cost(plan) for plan in plans]


def representative_gap_index(costs: list[GapRenderCost]) -> int:
    if not costs:
        raise ValueError("At least one Blender gap is required for runtime estimation")
    return max(costs, key=lambda item: (item.weight, -item.gap_index)).gap_index


def predicted_total_seconds(
    costs: list[GapRenderCost],
    representative_index: int,
    representative_elapsed_seconds: float,
) -> float:
    if representative_elapsed_seconds < 0.0:
        raise ValueError("Representative elapsed time cannot be negative")
    representative = next(
        (item for item in costs if item.gap_index == representative_index),
        None,
    )
    if representative is None:
        raise ValueError("Representative gap is not present in the render-cost list")
    return round(
        representative_elapsed_seconds
        * sum(item.weight for item in costs)
        / representative.weight,
        3,
    )


def enforce_runtime_budget(
    predicted_seconds: float,
    maximum_seconds: int,
    allow_override: bool,
) -> None:
    if predicted_seconds <= maximum_seconds or allow_override:
        return
    predicted_minutes = predicted_seconds / 60.0
    maximum_minutes = maximum_seconds / 60.0
    raise RenderRuntimeBudgetExceeded(
        "Predicted Blender render time "
        f"({predicted_minutes:.1f} minutes) exceeds the configured "
        f"{maximum_minutes:.1f}-minute runtime budget. "
        "The representative gap was saved; lower FPS, scale, samples, or entity count "
        "and resume, or explicitly enable the runtime-budget override."
    )


def preview_is_approved(approval_path: Path, signature: str) -> bool:
    try:
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("approved") is True
        and payload.get("signature") == signature
    )


def approve_representative_preview(approval_path: Path, signature: str) -> None:
    approval_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = approval_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps({"approved": True, "signature": signature}, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(approval_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _gap_render_cost(plan: dict) -> GapRenderCost:
    render_contract = plan.get("render", {})
    target_fps = min(
        float(plan["fps"]),
        float(render_contract.get("target_fps", 10)),
    )
    target_frames = max(2, round(float(plan["duration_seconds"]) * target_fps))
    fidelity_counts = _fidelity_counts(plan.get("entities", []))
    return GapRenderCost(
        gap_index=int(plan["gap_index"]),
        target_frames=target_frames,
        detailed_entities=fidelity_counts["detailed"],
        weak_entities=fidelity_counts["weak"],
    )


def _fidelity_counts(entities: list[dict]) -> dict[str, int]:
    weak_tiers = {"weak", "proxy"}
    weak_count = sum(
        str(entity.get("fidelity_tier", "weak")) in weak_tiers
        for entity in entities
    )
    return {
        "detailed": max(0, len(entities) - weak_count),
        "weak": weak_count,
    }
