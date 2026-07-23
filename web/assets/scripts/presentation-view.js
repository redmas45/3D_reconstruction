// @ts-check

/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @param {HTMLVideoElement} video
 * @returns {HTMLElement}
 */
export function createGapMarkerTrack(presentation, video) {
  const track = createElement("div", "gap-marker-track");
  track.setAttribute("aria-label", "Reconstructed interval markers");
  const duration = Math.max(0.001, presentation.source.duration_seconds);
  presentation.gaps.forEach((gap) => {
    const marker = /** @type {HTMLButtonElement} */ (
      createElement("button", "gap-marker")
    );
    marker.type = "button";
    marker.style.left = `${gap.start_seconds / duration * 100}%`;
    marker.style.width = `${Math.max(0.7, gap.duration_seconds / duration * 100)}%`;
    marker.title = `Review reconstructed gap ${gap.gap_index + 1} at ${formatTime(gap.start_seconds)}`;
    marker.setAttribute("aria-label", marker.title);
    marker.addEventListener("click", () => {
      video.currentTime = gap.start_seconds;
      void video.play();
    });
    track.append(marker);
  });
  return track;
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @param {HTMLVideoElement} video
 * @returns {HTMLElement}
 */
export function createPresentationView(presentation, video) {
  const view = createElement("div", "presentation-view");
  view.append(
    createStory(presentation),
    createClues(presentation),
    createGapReview(presentation, video),
  );
  return view;
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @returns {HTMLElement}
 */
function createStory(presentation) {
  const section = createElement("section", "presentation-story");
  section.append(
    createElement("span", "presentation-kicker", "Evidence-grounded story"),
    createElement("h4", "", presentation.story.headline),
    createElement("p", "presentation-summary", presentation.story.summary),
  );
  section.append(createStoryPoints(presentation.story.points || []));
  const metrics = createElement("div", "presentation-metrics");
  metrics.append(
    createMetric(`${Math.round(presentation.story.confidence * 100)}%`, "story confidence"),
    createMetric(`${Math.round(presentation.source.observed_fraction * 100)}%`, "visible evidence"),
    createMetric(String(presentation.gaps.length), "reconstructed gaps"),
    createMetric(presentation.render.engine || "3D", "render engine"),
  );
  section.append(metrics);
  section.append(createProvenance(presentation));
  if (presentation.story.warning) {
    section.append(createElement("p", "presentation-warning", presentation.story.warning));
  }
  section.append(createElement("p", "presentation-disclosure", presentation.disclosure));
  return section;
}


function createStoryPoints(points) {
  const list = createElement("ul", "presentation-story-points");
  points.slice(0, 5).forEach((point) => {
    list.append(createElement("li", "", point));
  });
  return list;
}


function createProvenance(presentation) {
  const provenance = createElement("div", "presentation-provenance");
  provenance.append(
    createBadge(`Planner · ${humanize(presentation.story.planning_mode || "unknown")}`),
    createBadge(`Model · ${presentation.story.deployment || "not reported"}`),
    createBadge(`Render · ${presentation.render.target_fps || "source"} fps`),
    createBadge(
      presentation.render.hybrid_static_backplate ? "Evidence backplate · on" : "Evidence backplate · off",
    ),
  );
  return provenance;
}


function createBadge(label) {
  return createElement("span", "presentation-badge", label);
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @returns {HTMLElement}
 */
function createClues(presentation) {
  const section = createElement("section", "presentation-clues");
  section.append(createElement("h4", "", "Strongest visible clues"));
  const list = createElement("ol", "");
  presentation.top_clues.slice(0, 5).forEach((clue) => {
    const item = createElement("li");
    const statement = createElement("span", "", clue.statement);
    const confidence = createElement(
      "small", "", `${Math.round(clue.confidence * 100)}%`,
    );
    item.append(statement, confidence);
    list.append(item);
  });
  if (!presentation.top_clues.length) {
    list.append(createElement("li", "empty-clue", "No ranked clue was available."));
  }
  section.append(list);
  return section;
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @param {HTMLVideoElement} video
 * @returns {HTMLElement}
 */
function createGapReview(presentation, video) {
  const section = createElement("section", "presentation-gaps");
  section.append(createElement("h4", "", "Review reconstructed intervals"));
  presentation.gaps.forEach((gap) => section.append(createGapCard(gap, video)));
  return section;
}


/**
 * @param {import("./api-client.js").PresentationGap} gap
 * @param {HTMLVideoElement} video
 * @returns {HTMLElement}
 */
function createGapCard(gap, video) {
  const card = createElement("article", "presentation-gap-card");
  card.append(createGapHeader(gap, video), createGapMetrics(gap));
  const phases = createElement("div", "presentation-gap-phases");
  phases.append(
    createPhase("Before · observed", gap.before_observed),
    createPhase("Inside · inferred", gap.inside_inferred, true),
    createPhase("After · observed", gap.after_observed),
  );
  card.append(phases, createDecisionTrace(gap));
  return card;
}


function createGapHeader(gap, video) {
  const header = createElement("div", "presentation-gap-heading");
  const seek = /** @type {HTMLButtonElement} */ (
    createElement(
      "button",
      "gap-seek",
      `Gap ${gap.gap_index + 1} · ${formatTime(gap.start_seconds)} · ${gap.duration_seconds.toFixed(1)}s`,
    )
  );
  seek.type = "button";
  seek.addEventListener("click", () => {
    video.currentTime = gap.start_seconds;
    void video.play();
  });
  header.append(
    seek,
    createElement("span", "", `${Math.round(gap.confidence * 100)}% confidence`),
  );
  return header;
}


function createGapMetrics(gap) {
  const clues = gap.clues || [];
  const metrics = createElement("div", "presentation-gap-metrics");
  metrics.append(
    createMetric(String(gap.entity_count ?? 0), "supported entities"),
    createMetric(
      `${Math.round((gap.calibration_confidence || 0) * 100)}%`,
      "calibration confidence",
    ),
    createMetric(String(clues.length), "linked clues"),
  );
  return metrics;
}


function createDecisionTrace(gap) {
  const details = /** @type {HTMLDetailsElement} */ (
    createElement("details", "presentation-decision-trace")
  );
  details.open = gap.gap_index === 0;
  details.append(createElement("summary", "", "Evidence and decision trace"));
  const content = createElement("div", "presentation-trace-content");
  content.append(
    createGapClues(gap.clues || []),
    createEventBeats(gap.event_beats || []),
    createEntityDecisions(gap.entities || []),
  );
  if ((gap.evidence_references || []).length) {
    content.append(createTextList(
      "Evidence references", gap.evidence_references, "presentation-references",
    ));
  }
  if (gap.unknowns.length) {
    content.append(createTextList("Remaining unknowns", gap.unknowns, "presentation-unknowns"));
  }
  details.append(content);
  return details;
}


function createEventBeats(beats) {
  if (!beats.length) {
    return createElement("span", "");
  }
  const section = createElement("section", "presentation-trace-section");
  section.append(createElement("h5", "", "Inferred motion sequence"));
  const sequence = createElement("div", "presentation-event-beats");
  beats.forEach((beat) => {
    const entityLabel = beat.entity_ids.map(humanize).join(", ") || "scene";
    sequence.append(createBadge(
      `${Math.round(beat.time_fraction * 100)}% · ${humanize(beat.action)} · ${entityLabel}`,
    ));
  });
  section.append(sequence);
  return section;
}


function createGapClues(clues) {
  if (!clues.length) {
    return createElement("p", "presentation-empty", "No gap-specific clue was published.");
  }
  const section = createElement("section", "presentation-trace-section");
  section.append(createElement("h5", "", "Visible clues used"));
  const list = createElement("ul", "presentation-trace-list");
  clues.forEach((clue) => {
    list.append(createElement(
      "li", "", `${clue.statement} · ${Math.round(clue.confidence * 100)}%`,
    ));
  });
  section.append(list);
  return section;
}


function createEntityDecisions(entities) {
  const section = createElement("section", "presentation-trace-section");
  section.append(createElement("h5", "", "Hypothesis decisions"));
  if (!entities.length) {
    section.append(createElement("p", "presentation-empty", "No supported entity decision."));
    return section;
  }
  entities.forEach((entity) => section.append(createEntityDecision(entity)));
  return section;
}


function createEntityDecision(entity) {
  const card = createElement("article", "presentation-entity-decision");
  const heading = createElement("div", "presentation-entity-heading");
  heading.append(
    createElement("strong", "", humanize(entity.entity_id)),
    createElement("span", "", `${Math.round(entity.confidence * 100)}%`),
  );
  card.append(
    heading,
    createElement("p", "presentation-selected", `Selected · ${humanize(entity.selected_hypothesis_id)}`),
    createElement("p", "presentation-rationale", entity.decision_summary),
  );
  if (entity.rejected_hypotheses.length) {
    card.append(createRejectedHypotheses(entity.rejected_hypotheses));
  }
  return card;
}


function createRejectedHypotheses(rejections) {
  const details = /** @type {HTMLDetailsElement} */ (
    createElement("details", "presentation-rejections")
  );
  details.append(createElement("summary", "", `${rejections.length} rejected alternative(s)`));
  const list = createElement("ul", "presentation-trace-list");
  rejections.forEach((item) => {
    list.append(createElement("li", "", `${humanize(item.id)} — ${item.reason}`));
  });
  details.append(list);
  return details;
}


function createTextList(title, items, className) {
  const section = createElement("section", "presentation-trace-section");
  section.append(createElement("h5", "", title));
  const list = createElement("ul", `presentation-trace-list ${className}`);
  items.forEach((item) => list.append(createElement("li", "", item)));
  section.append(list);
  return section;
}


function createPhase(label, statement, inferred = false) {
  const phase = createElement(
    "div", `presentation-gap-phase${inferred ? " inferred" : ""}`,
  );
  phase.append(
    createElement("small", "", label),
    createElement("p", "", statement),
  );
  return phase;
}


function createMetric(value, label) {
  const metric = createElement("span", "presentation-metric");
  metric.append(
    createElement("strong", "", value),
    createElement("small", "", label),
  );
  return metric;
}


function formatTime(totalSeconds) {
  const boundedSeconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(boundedSeconds / 60);
  const seconds = boundedSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}


function humanize(value) {
  return String(value).replaceAll("_", " ");
}


function createElement(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}
