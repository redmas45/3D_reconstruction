// @ts-check

const FRAME_CAPTURE_WIDTH = 640;
const FRAME_CAPTURE_HEIGHT = 360;
const BOUNDARY_REVIEW_OFFSET_SECONDS = 0.15;
const MAXIMUM_VISIBLE_STORY_POINTS = 3;
const MAXIMUM_VISIBLE_CLUES = 3;


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
    marker.title = `Review gap ${gap.gap_index + 1} at ${formatTime(gap.start_seconds)}`;
    marker.setAttribute("aria-label", marker.title);
    marker.addEventListener("click", () => seekVideo(video, gap.start_seconds));
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
    createGapWorkspace(presentation, video),
  );
  return view;
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @returns {HTMLElement}
 */
function createStory(presentation) {
  const section = createElement("section", "presentation-story");
  const heading = createElement("div", "presentation-section-heading");
  heading.append(
    createElement("span", "presentation-kicker", "Evidence-grounded finding"),
    createElement("span", "confidence-pill", confidenceLabel(presentation.story.confidence)),
  );
  section.append(
    heading,
    createElement("h4", "", presentation.story.headline),
    createElement("p", "presentation-summary", presentation.story.summary),
    createStoryMetrics(presentation),
  );
  const points = createVisibleStoryPoints(presentation);
  if (points.childElementCount) section.append(points);
  if (presentation.story.warning) {
    section.append(createElement("p", "presentation-warning", presentation.story.warning));
  }
  section.append(createElement("p", "presentation-disclosure", presentation.disclosure));
  return section;
}


function createStoryMetrics(presentation) {
  const metrics = createElement("div", "presentation-metrics");
  const reconstructedFraction = presentation.output?.reconstructed_fraction
    ?? 1 - presentation.source.observed_fraction;
  metrics.append(
    createMetric(`${Math.round(presentation.source.observed_fraction * 100)}%`, "visible evidence"),
    createMetric(`${Math.round(reconstructedFraction * 100)}%`, "reconstructed"),
    createMetric(String(presentation.gaps.length), "reviewable gaps"),
  );
  return metrics;
}


function createVisibleStoryPoints(presentation) {
  const list = createElement("ul", "presentation-story-points");
  const storyPoints = presentation.story.points || [];
  const statements = storyPoints.length
    ? storyPoints
    : presentation.top_clues.map((clue) => clue.statement);
  statements.slice(0, MAXIMUM_VISIBLE_STORY_POINTS).forEach((statement) => {
    list.append(createElement("li", "", statement));
  });
  return list;
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @param {HTMLVideoElement} video
 * @returns {HTMLElement}
 */
function createGapWorkspace(presentation, video) {
  const section = createElement("section", "presentation-gap-workspace");
  const heading = createElement("div", "presentation-section-heading");
  heading.append(
    createElement("div", "", ""),
    createElement("span", "presentation-kicker", "Gap review"),
  );
  heading.firstElementChild?.append(
    createElement("h4", "", "Inspect the reconstruction"),
    createElement("p", "presentation-helper", "Select a gap to compare the evidence boundaries and inferred interval."),
  );
  const tabs = createElement("div", "gap-selector");
  tabs.setAttribute("role", "tablist");
  const detail = createElement("div", "gap-review-detail");
  const tabButtons = presentation.gaps.map((gap, index) => {
    const button = createGapTab(gap, index === 0);
    button.addEventListener("click", () => {
      setActiveGapTab(tabButtons, button);
      renderSelectedGap(detail, presentation, gap, video);
    });
    tabs.append(button);
    return button;
  });
  section.append(heading, tabs, detail);
  if (presentation.gaps.length) {
    renderSelectedGap(detail, presentation, presentation.gaps[0], video);
  } else {
    detail.append(createElement("p", "presentation-empty", "No reconstructed interval was published."));
  }
  return section;
}


function createGapTab(gap, selected) {
  const button = /** @type {HTMLButtonElement} */ (
    createElement("button", `gap-tab${selected ? " active" : ""}`)
  );
  button.type = "button";
  button.setAttribute("role", "tab");
  button.setAttribute("aria-selected", String(selected));
  button.append(
    createElement("strong", "", `Gap ${gap.gap_index + 1}`),
    createElement("span", "", `${formatTime(gap.start_seconds)} · ${gap.duration_seconds.toFixed(1)}s`),
  );
  return button;
}


function setActiveGapTab(buttons, selectedButton) {
  buttons.forEach((button) => {
    const selected = button === selectedButton;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
}


function renderSelectedGap(detail, presentation, gap, video) {
  const frameReview = createFrameReview(gap, video);
  const explanation = createGapExplanation(gap);
  const technicalAudit = createTechnicalAudit(gap, presentation);
  detail.replaceChildren(
    createGapHeading(gap, video),
    frameReview.element,
    explanation,
    technicalAudit,
  );
  void captureGapFrames(video, gap, frameReview.canvases)
    .catch(() => markCaptureUnavailable(frameReview.element));
}


function createGapHeading(gap, video) {
  const header = createElement("div", "gap-review-heading");
  const title = createElement("div");
  title.append(
    createElement("h5", "", `Gap ${gap.gap_index + 1} reconstruction`),
    createElement("p", "", `${formatTime(gap.start_seconds)}–${formatTime(gap.end_seconds)} · ${gap.duration_seconds.toFixed(1)} seconds`),
  );
  const play = /** @type {HTMLButtonElement} */ (
    createElement("button", "review-play-button", "Play this gap")
  );
  play.type = "button";
  play.addEventListener("click", () => seekVideo(video, gap.start_seconds));
  header.append(title, createElement("span", "confidence-pill", confidenceLabel(gap.confidence)), play);
  return header;
}


function createFrameReview(gap, video) {
  const review = createElement("div", "gap-frame-review");
  const phases = [
    ["Before", "Observed evidence", gap.before_observed, Math.max(0, gap.start_seconds - BOUNDARY_REVIEW_OFFSET_SECONDS)],
    ["Reconstruction", "AI-inferred interval", gap.inside_inferred, gap.start_seconds + gap.duration_seconds / 2],
    ["After", "Observed evidence", gap.after_observed, gap.end_seconds + BOUNDARY_REVIEW_OFFSET_SECONDS],
  ];
  const canvases = phases.map(([label, state, description, time], index) => {
    const frame = createFrameCard(label, state, description, index === 1);
    frame.element.addEventListener("click", () => seekVideo(video, Number(time)));
    review.append(frame.element);
    return frame.canvas;
  });
  return { element: review, canvases };
}


function createFrameCard(label, state, description, inferred) {
  const button = /** @type {HTMLButtonElement} */ (
    createElement("button", `evidence-frame${inferred ? " inferred" : ""}`)
  );
  button.type = "button";
  const canvas = /** @type {HTMLCanvasElement} */ (document.createElement("canvas"));
  canvas.width = FRAME_CAPTURE_WIDTH;
  canvas.height = FRAME_CAPTURE_HEIGHT;
  canvas.setAttribute("aria-label", `${label} video frame`);
  const copy = createElement("span", "evidence-frame-copy");
  copy.append(
    createElement("small", "", state),
    createElement("strong", "", label),
    createElement("span", "", description),
  );
  button.append(canvas, copy);
  return { element: button, canvas };
}


function createGapExplanation(gap) {
  const layout = createElement("div", "gap-explanation");
  const decision = createElement("section", "gap-decision");
  decision.append(
    createElement("span", "presentation-kicker", "Reconstruction decision"),
    createElement("p", "", gap.inside_inferred),
  );
  const clues = createElement("section", "gap-clues");
  clues.append(createElement("span", "presentation-kicker", "Evidence used"));
  const list = createElement("ul");
  (gap.clues || []).slice(0, MAXIMUM_VISIBLE_CLUES).forEach((clue) => {
    list.append(createElement("li", "", clue.statement));
  });
  if (!list.childElementCount) {
    list.append(createElement("li", "", "Boundary detections and measured motion continuity."));
  }
  clues.append(list);
  layout.append(decision, clues);
  if (gap.unknowns.length) {
    layout.append(createElement("p", "gap-primary-uncertainty", `Key uncertainty · ${gap.unknowns[0]}`));
  }
  return layout;
}


function createTechnicalAudit(gap, presentation) {
  const details = /** @type {HTMLDetailsElement} */ (
    createElement("details", "presentation-technical-audit")
  );
  details.append(createElement("summary", "", "Technical audit"));
  const content = createElement("div", "technical-audit-content");
  content.append(createAuditMetrics(gap, presentation));
  if (gap.entities.length) content.append(createEntityDecisions(gap.entities));
  if (gap.evidence_references.length) {
    content.append(createTextList("Evidence references", gap.evidence_references));
  }
  if (gap.unknowns.length > 1) {
    content.append(createTextList("Additional unknowns", gap.unknowns.slice(1)));
  }
  details.append(content);
  return details;
}


function createAuditMetrics(gap, presentation) {
  const metrics = createElement("div", "technical-metrics");
  metrics.append(
    createMetric(String(gap.entity_count ?? 0), "rendered entities"),
    createMetric(`${Math.round((gap.calibration_confidence || 0) * 100)}%`, "camera calibration"),
    createMetric(presentation.render.engine || "3D", "render engine"),
  );
  return metrics;
}


function createEntityDecisions(entities) {
  const section = createElement("section", "technical-section");
  section.append(createElement("h6", "", "Entity decisions"));
  entities.forEach((entity) => {
    const card = createElement("article", "technical-entity");
    card.append(
      createElement("strong", "", humanize(entity.entity_id)),
      createElement("span", "", `${Math.round(entity.confidence * 100)}% · ${humanize(entity.selected_hypothesis_id)}`),
      createElement("p", "", entity.decision_summary),
    );
    if (entity.rejected_hypotheses.length) {
      const rejected = entity.rejected_hypotheses.map((item) => humanize(item.id)).join(", ");
      card.append(createElement("small", "", `Rejected: ${rejected}`));
    }
    section.append(card);
  });
  return section;
}


function createTextList(title, items) {
  const section = createElement("section", "technical-section");
  section.append(createElement("h6", "", title));
  const list = createElement("ul");
  items.forEach((item) => list.append(createElement("li", "", item)));
  section.append(list);
  return section;
}


async function captureGapFrames(video, gap, canvases) {
  const source = video.currentSrc || video.getAttribute("src") || "";
  if (!source) throw new Error("Video source is unavailable");
  const sampler = document.createElement("video");
  sampler.preload = "auto";
  sampler.muted = true;
  sampler.src = source;
  try {
    await waitForMediaEvent(sampler, "loadedmetadata");
    const times = [
      Math.max(0, gap.start_seconds - BOUNDARY_REVIEW_OFFSET_SECONDS),
      gap.start_seconds + gap.duration_seconds / 2,
      Math.min(sampler.duration, gap.end_seconds + BOUNDARY_REVIEW_OFFSET_SECONDS),
    ];
    for (let index = 0; index < canvases.length; index += 1) {
      await captureFrame(sampler, times[index], canvases[index]);
    }
  } finally {
    sampler.removeAttribute("src");
    sampler.load();
  }
}


async function captureFrame(sampler, time, canvas) {
  sampler.currentTime = Math.max(0, Math.min(time, sampler.duration || time));
  await waitForMediaEvent(sampler, "seeked");
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable");
  context.drawImage(sampler, 0, 0, canvas.width, canvas.height);
}


function waitForMediaEvent(media, eventName) {
  if (eventName === "loadedmetadata" && media.readyState >= HTMLMediaElement.HAVE_METADATA) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const onSuccess = () => {
      media.removeEventListener("error", onError);
      resolve(undefined);
    };
    const onError = () => {
      media.removeEventListener(eventName, onSuccess);
      reject(new Error(`Video ${eventName} failed`));
    };
    media.addEventListener(eventName, onSuccess, { once: true });
    media.addEventListener("error", onError, { once: true });
  });
}


function markCaptureUnavailable(container) {
  container.classList.add("capture-unavailable");
}


function seekVideo(video, seconds) {
  video.currentTime = Math.max(0, seconds);
  void video.play();
}


function createMetric(value, label) {
  const metric = createElement("span", "presentation-metric");
  metric.append(
    createElement("strong", "", value),
    createElement("small", "", label),
  );
  return metric;
}


function confidenceLabel(confidence) {
  return `${Math.round(confidence * 100)}% confidence`;
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
