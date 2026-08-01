// @ts-check
// The judge-facing reconstruction dashboard for one completed job, built entirely from a
// validated scene manifest. It presents the observed evidence, the inferred continuation,
// confidence, and unresolved uncertainty in plain language, and hosts the interactive
// Three.js viewer with its controls. The 3D viewer is created lazily (on first open) and
// disposed on close, so a page listing several completed jobs holds at most one live
// WebGL context.

import { ReconstructionView } from "../three/three-reconstruction-view.js";
import { supportTier } from "../three/confidence-visuals.js";

const CONFIDENCE_WORDS = { strong: "Well supported", mixed: "Mixed support", weak: "Weak support" };

/**
 * @param {object} scene a validated scene manifest
 * @param {{ sourceName?: string }} [options]
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function createReconstructionDashboard(scene, options = {}) {
  const manifest = /** @type {any} */ (scene);
  /** @type {ReconstructionView | null} */
  let view = null;
  let gapIndex = 0;

  const root = el("section", "recon-dashboard");
  root.append(
    buildOverview(manifest, options.sourceName),
    buildEvidenceMap(manifest),
    buildSummary(manifest),
    buildClues(manifest),
    buildDecisionTrace(manifest),
  );

  const stage = el("div", "recon-stage");
  const legend = el("div", "recon-legend");
  const gapDetail = el("div", "recon-gap-detail");
  const controls = el("div", "recon-controls");

  const viewer = el("div", "recon-viewer");
  viewer.append(stage, controls);
  root.append(sectionHeading("Interactive 3D reconstruction",
    "A transparent visualization of the inferred motion. It is a hypothesis, not recovered footage."));
  root.append(viewer, legend, gapDetail);

  function renderGap(index) {
    gapIndex = index;
    legend.replaceChildren(buildLegend(manifest.gaps[index]));
    gapDetail.replaceChildren(buildGapDetail(manifest.gaps[index]));
    if (view) view.selectGap(index);
  }

  function mountViewer() {
    view = new ReconstructionView(stage);
    view.load(manifest);
    if (!view.isAvailable) {
      controls.replaceChildren(el("p", "recon-note",
        "The 3D viewer needs WebGL, which is unavailable here. The written analysis above remains complete."));
      return;
    }
    controls.replaceChildren(...buildControls(manifest, {
      onGap: renderGap,
      getGapIndex: () => gapIndex,
      view,
    }));
  }

  // Lazy mount: build the viewer only when the user asks for it.
  const openButton = /** @type {HTMLButtonElement} */ (el("button", "recon-open", "▶ Open 3D reconstruction"));
  openButton.type = "button";
  openButton.addEventListener("click", () => {
    openButton.remove();
    mountViewer();
    renderGap(0);
  }, { once: true });
  controls.append(openButton);
  legend.replaceChildren(buildLegend(manifest.gaps[0] || { entities: [] }));
  gapDetail.replaceChildren(buildGapDetail(manifest.gaps[0] || {}));

  return {
    element: root,
    dispose() { if (view) view.dispose(); view = null; },
  };
}

function buildOverview(manifest, sourceName) {
  const source = manifest.source || {};
  const section = el("section", "recon-overview");
  section.append(el("span", "recon-eyebrow", "Reconstruction report"));
  section.append(el("h2", "", (sourceName || source.video_name || "Video") + " — reconstructed"));
  section.append(el("p", "recon-disclosure", manifest.evidence_disclosure || ""));
  const metrics = el("div", "recon-metrics");
  metrics.append(
    metric(formatDuration(source.duration_seconds), "source duration"),
    metric(pct(source.observed_fraction), "visible evidence"),
    metric(pct(source.reconstructed_fraction), "inferred"),
    metric(String((manifest.gaps || []).length), "missing intervals"),
    metric("Three.js", "renderer"),
  );
  section.append(metrics);
  return section;
}

function buildSummary(manifest) {
  const narrative = manifest.narrative || {};
  const section = el("section", "recon-summary");
  const head = el("div", "recon-summary-head");
  head.append(el("h3", "", narrative.headline || "Inferred summary"));
  head.append(el("span", "recon-pill " + tierClass(narrative.confidence), confidenceLabel(narrative.confidence)));
  section.append(head);
  section.append(el("p", "recon-summary-body", narrative.whole_video_summary || ""));
  section.append(el("small", narrative.causal_link_supported ? "recon-causal ok" : "recon-causal limited",
    narrative.causal_link_supported
      ? "Causal links are evidence-supported."
      : "Sequence is inferred; causal links are not claimed."));
  if ((narrative.story_points || []).length) {
    const list = el("ul", "recon-points");
    narrative.story_points.forEach((point) => list.append(el("li", "", point)));
    section.append(list);
  }
  if ((narrative.unknowns || []).length) {
    section.append(el("p", "recon-unknowns", "Unresolved uncertainty · " + narrative.unknowns.join(" · ")));
  }
  return section;
}

function buildEvidenceMap(manifest) {
  const source = manifest.source || {};
  const totalSeconds = Math.max(0.001, Number(source.duration_seconds) || 0);
  const section = el("section", "recon-evidence-map");
  section.append(sectionHeading("Evidence map", "The visible 75% stays observed; only the marked intervals are inferred."));
  const track = el("div", "evidence-map-track");
  let cursor = 0;
  (manifest.gaps || []).forEach((gap) => {
    const start = Math.max(0, Number(gap.start_seconds) || 0);
    const end = Math.max(start, Number(gap.end_seconds) || start);
    appendMapSegment(track, "observed", start - cursor, totalSeconds);
    appendMapSegment(track, "inferred", end - start, totalSeconds, `Gap ${Number(gap.gap_index) + 1}`);
    cursor = end;
  });
  appendMapSegment(track, "observed", totalSeconds - cursor, totalSeconds);
  section.append(track);
  const legend = el("div", "evidence-map-legend");
  legend.append(legendItem("observed", "Visible evidence"), legendItem("inferred", "Inferred continuation"));
  section.append(legend);
  return section;
}

function appendMapSegment(track, state, seconds, totalSeconds, label = "") {
  if (seconds <= 0) return;
  const segment = el("span", `evidence-map-segment ${state}`);
  segment.style.flex = `${seconds / totalSeconds} 1 0`;
  if (label) segment.title = `${label} · ${seconds.toFixed(1)}s`;
  track.append(segment);
}

function legendItem(state, label) {
  const item = el("span", "evidence-map-legend-item");
  item.append(el("i", `evidence-map-swatch ${state}`), el("span", "", label));
  return item;
}

function buildDecisionTrace(manifest) {
  const details = /** @type {HTMLDetailsElement} */ (el("details", "recon-decision-trace"));
  const summary = el("summary");
  summary.append(el("strong", "", "How each gap was chosen"));
  summary.append(el("span", "recon-trace-count", `${(manifest.gaps || []).length} bounded decisions`));
  details.append(summary);
  const list = el("div", "recon-trace-list");
  (manifest.gaps || []).forEach((gap) => {
    const item = el("article", "recon-trace-item");
    const decision = gap.decision || {};
    const refs = (decision.evidence_references || []).slice(0, 4);
    item.append(el("div", "recon-trace-heading", `Gap ${Number(gap.gap_index) + 1} · ${pct(decision.confidence || gap.confidence)} support`));
    item.append(el("p", "", decision.gap_summary || gap.narrative?.inside_inferred || "Continuation bounded by visible boundaries."));
    item.append(traceRow("Evidence used", refs.length ? refs.join(" · ") : "Boundary motion and track continuity"));
    item.append(traceRow("Unknowns", (decision.unknowns || gap.narrative?.unknowns || []).join(" · ") || "None recorded"));
    list.append(item);
  });
  if (!list.childElementCount) list.append(el("p", "recon-note", "Decision details will appear after reasoning completes."));
  details.append(list);
  return details;
}

function traceRow(label, value) {
  const row = el("div", "recon-trace-row");
  row.append(el("small", "", label), el("span", "", value));
  return row;
}

function buildClues(manifest) {
  const section = el("section", "recon-clues");
  section.append(sectionHeading("Observed evidence clues", "Facts read from the visible footage."));
  const list = el("ul", "recon-clue-list");
  (manifest.clues || []).forEach((clue) => {
    const item = el("li", "recon-clue");
    item.append(el("span", "recon-clue-cat", humanize(clue.category)));
    item.append(el("strong", "", clue.statement));
    item.append(el("span", "recon-clue-conf", pct(clue.confidence)));
    list.append(item);
  });
  if (!list.childElementCount) list.append(el("li", "recon-clue", "Boundary detections and measured motion continuity."));
  section.append(list);
  return section;
}

function buildControls(manifest, context) {
  const { view, onGap, getGapIndex } = context;
  const bar = el("div", "recon-control-bar");
  const gaps = manifest.gaps || [];

  const selector = el("div", "recon-gap-tabs");
  gaps.forEach((gap, index) => {
    const tab = /** @type {HTMLButtonElement} */ (el("button", "recon-gap-tab" + (index === 0 ? " active" : ""),
      `Gap ${index + 1} · ${(gap.duration_seconds || 0).toFixed(1)}s`));
    tab.type = "button";
    tab.addEventListener("click", () => {
      selector.querySelectorAll(".recon-gap-tab").forEach((node) => node.classList.remove("active"));
      tab.classList.add("active");
      onGap(index);
    });
    selector.append(tab);
  });

  const play = button("Pause", () => {
    if (view._playing) { view.pause(); play.textContent = "Play"; }
    else { view.play(); play.textContent = "Pause"; }
  });
  const buttons = el("div", "recon-buttons");
  buttons.append(
    play,
    button("Restart", () => view.restart()),
    toggle("Uncertainty", true, (on) => view.setUncertainty(on)),
    toggle("Debug grid", false, (on) => view.setDebug(on)),
    captureButton(view),
  );
  bar.append(selector, buttons);
  return [bar];
}

function captureButton(view) {
  const btn = button("● Capture clip", async () => {
    if (!view.captureSupported()) { btn.textContent = "Capture unsupported"; btn.disabled = true; return; }
    if (btn.dataset.recording === "1") {
      btn.dataset.recording = "0";
      btn.textContent = "Saving…";
      await view.stopCapture("reconstruction-preview.webm");
      btn.textContent = "● Capture clip";
      return;
    }
    view.startCapture();
    btn.dataset.recording = "1";
    btn.textContent = "■ Stop & save";
  });
  return btn;
}

function buildLegend(gap) {
  const wrap = el("div", "recon-legend-inner");
  (gap.entities || []).forEach((entity) => {
    const chip = el("span", "recon-chip " + tierClass(entity.confidence));
    chip.append(el("span", "recon-chip-dot"));
    chip.append(el("strong", "", `${humanize(entity.class_name)} ${entity.track_id}`));
    chip.append(el("span", "", `${pct(entity.confidence)} · ${humanize(entity.visual_fidelity_tier)}`));
    wrap.append(chip);
  });
  if (!wrap.childElementCount) wrap.append(el("span", "recon-note", "No entities were confidently placed in this interval."));
  return wrap;
}

function buildGapDetail(gap) {
  const wrap = el("div", "recon-gap-inner");
  if (!gap || gap.gap_index === undefined) return wrap;
  const narrative = gap.narrative || {};
  const phases = el("div", "recon-phases");
  phases.append(
    phase("Before · observed", narrative.before_observed),
    phase("Inside · inferred", narrative.inside_inferred),
    phase("After · observed", narrative.after_observed),
  );
  wrap.append(phases);
  (gap.entities || []).forEach((entity) => wrap.append(entityDecision(entity)));
  if ((narrative.unknowns || []).length) {
    wrap.append(el("p", "recon-unknowns", "Unresolved uncertainty · " + narrative.unknowns.join(" · ")));
  }
  return wrap;
}

function entityDecision(entity) {
  const card = el("article", "recon-decision");
  const head = el("div", "recon-decision-head");
  head.append(el("strong", "", `${humanize(entity.class_name)} ${entity.track_id}`));
  head.append(el("span", "recon-pill " + tierClass(entity.confidence), pct(entity.confidence)));
  card.append(head);
  card.append(el("p", "", "Selected: " + humanize(shortHypothesis(entity))));
  const selection = entity.selected_hypothesis || {};
  if (Number.isFinite(Number(selection.selection_score)) && Number(selection.selection_score) > 0) {
    const source = selection.selection_source === "safety_gate"
      ? "safety gate"
      : selection.selection_source === "deterministic_ranker" ? "measured ranker" : "Azure ranker";
    card.append(el("small", "recon-decision-score",
      `Measured fit ${Math.round(Number(selection.selection_score) * 100)}% · ${source}`));
  }
  if (entity.decision_summary) card.append(el("p", "recon-decision-summary", entity.decision_summary));
  if ((entity.rejected_hypotheses || []).length) {
    card.append(el("small", "recon-rejected",
      "Rejected: " + entity.rejected_hypotheses.map((item) => humanize(shortId(item.id, entity.track_id))).join(", ")));
  }
  const disagreement = entity.uncertainty?.heading_disagreement_degrees || 0;
  if (disagreement > 45) {
    card.append(el("small", "recon-warn",
      `Boundary headings disagree by ${Math.round(disagreement)}°; the path is held uncertain rather than forced smooth.`));
  }
  return card;
}

// ---- small helpers -------------------------------------------------------

function shortHypothesis(entity) {
  return shortId(entity.selected_hypothesis?.id || "", entity.track_id) || entity.selected_hypothesis?.type || "continuation";
}

function shortId(id, trackId) {
  const marker = `${trackId}_`;
  const tail = String(id).split(marker).pop();
  return tail || String(id);
}

function sectionHeading(title, note) {
  const head = el("div", "recon-heading");
  head.append(el("h3", "", title));
  if (note) head.append(el("p", "recon-helper", note));
  return head;
}

function metric(value, label) {
  const node = el("span", "recon-metric");
  node.append(el("strong", "", value), el("small", "", label));
  return node;
}

function phase(label, text) {
  const node = el("div", "recon-phase");
  node.append(el("small", "", label), el("p", "", text || "—"));
  return node;
}

function button(label, handler) {
  const btn = /** @type {HTMLButtonElement} */ (el("button", "recon-btn", label));
  btn.type = "button";
  btn.addEventListener("click", handler);
  return btn;
}

function toggle(label, initial, handler) {
  let on = initial;
  const btn = button((initial ? "✓ " : "○ ") + label, () => {
    on = !on;
    btn.textContent = (on ? "✓ " : "○ ") + label;
    handler(on);
  });
  return btn;
}

function tierClass(confidence) {
  return "tier-" + supportTier(Number(confidence) || 0);
}

function confidenceLabel(confidence) {
  const value = Number(confidence) || 0;
  return `${CONFIDENCE_WORDS[supportTier(value)]} · ${Math.round(value * 100)}%`;
}

function pct(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  return minutes ? `${minutes}m ${total % 60}s` : `${total}s`;
}

function humanize(value) {
  return String(value || "").replaceAll("_", " ");
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
