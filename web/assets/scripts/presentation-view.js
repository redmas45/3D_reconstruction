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
  const metrics = createElement("div", "presentation-metrics");
  metrics.append(
    createMetric(`${Math.round(presentation.story.confidence * 100)}%`, "story confidence"),
    createMetric(`${Math.round(presentation.source.observed_fraction * 100)}%`, "visible evidence"),
    createMetric(String(presentation.gaps.length), "reconstructed gaps"),
  );
  section.append(metrics);
  section.append(createElement("p", "presentation-disclosure", presentation.disclosure));
  return section;
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
  const phases = createElement("div", "presentation-gap-phases");
  phases.append(
    createPhase("Before · observed", gap.before_observed),
    createPhase("Inside · inferred", gap.inside_inferred, true),
    createPhase("After · observed", gap.after_observed),
  );
  card.append(header, phases);
  if (gap.unknowns.length) {
    card.append(createElement(
      "small", "presentation-unknowns", `Unknown: ${gap.unknowns.join(" · ")}`,
    ));
  }
  return card;
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


function createElement(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}
