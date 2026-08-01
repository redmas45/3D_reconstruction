// @ts-check

const MAXIMUM_VISIBLE_STORY_POINTS = 3;
const MAXIMUM_VISIBLE_CLUES = 5;


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @returns {HTMLElement}
 */
export function createPresentationOverview(presentation) {
  const section = createElement("section", "case-overview");
  const heading = createElement("div", "presentation-section-heading");
  heading.append(
    createElement("span", "presentation-kicker", "Full-video finding"),
    createElement(
      "span", "confidence-pill",
      `${Math.round(presentation.story.confidence * 100)}% confidence`,
    ),
  );
  section.append(
    heading,
    createElement("h2", "", presentation.story.headline),
    createElement("p", "case-summary", presentation.story.summary),
    createCaseMetrics(presentation),
  );
  const points = createStoryPoints(presentation);
  if (points.childElementCount) section.append(points);
  if (presentation.story.warning) {
    section.append(createElement("p", "presentation-warning", presentation.story.warning));
  }
  section.append(createDisclosure(presentation.disclosure));
  return section;
}


/**
 * @param {import("./api-client.js").PresentationManifest} presentation
 * @returns {HTMLElement}
 */
export function createEvidenceNarrative(presentation) {
  const section = createElement("section", "evidence-narrative");
  section.append(createSectionHeading());
  const grid = createElement("div", "evidence-narrative-grid");
  grid.append(
    createObservedEvidence(presentation),
    createClueLedger(presentation),
  );
  section.append(grid, createDecisionPipeline(presentation));
  return section;
}


function createCaseMetrics(presentation) {
  const observedFraction = presentation.source.observed_fraction;
  const missingFraction = presentation.output?.reconstructed_fraction ?? 1 - observedFraction;
  const metrics = createElement("div", "case-metrics");
  metrics.append(
    createMetric(formatDuration(presentation.source.duration_seconds), "source duration"),
    createMetric(`${Math.round(observedFraction * 100)}%`, "visible evidence"),
    createMetric(`${Math.round(missingFraction * 100)}%`, "missing and reconstructed"),
    createMetric(String(presentation.gaps.length), "patched intervals"),
  );
  return metrics;
}


function createStoryPoints(presentation) {
  const list = createElement("ul", "case-story-points");
  const points = presentation.story.points?.length
    ? presentation.story.points
    : presentation.top_clues.map((clue) => clue.statement);
  points.slice(0, MAXIMUM_VISIBLE_STORY_POINTS).forEach((point) => {
    list.append(createElement("li", "", point));
  });
  return list;
}


function createDisclosure(disclosure) {
  const note = createElement("div", "evidence-disclosure");
  note.append(
    createElement("strong", "", "Evidence boundary"),
    createElement("p", "", disclosure),
  );
  return note;
}


function createSectionHeading() {
  const heading = createElement("div", "evidence-section-heading");
  const copy = createElement("div");
  copy.append(
    createElement("span", "presentation-kicker", "From evidence to reconstruction"),
    createElement("h3", "", "What the system saw and how it made the patch"),
    createElement(
      "p", "presentation-helper",
      "The explanation below separates direct observations from inferred decisions.",
    ),
  );
  heading.append(copy, createElement("span", "audit-badge", "Auditable"));
  return heading;
}


function createObservedEvidence(presentation) {
  const overview = presentation.evidence_overview;
  const observedSeconds = overview?.observed_seconds
    ?? presentation.source.duration_seconds * presentation.source.observed_fraction;
  const card = createElement("article", "narrative-card observed-evidence-card");
  card.append(
    createElement("span", "card-label", "Observed 75%"),
    createElement("h4", "", "What remained visible"),
    createElement(
      "p", "narrative-card-summary",
      overview?.summary ?? "The system analyzed visible footage around every missing interval.",
    ),
  );
  const metrics = createElement("div", "evidence-facts");
  metrics.append(
    createFact(formatDuration(observedSeconds), "analyzed"),
    createFact(String(overview?.tracked_entity_count ?? "—"), "tracked entities"),
    createFact(String(overview?.clue_count ?? presentation.top_clues.length), "recorded clues"),
  );
  card.append(metrics);
  return card;
}


function createClueLedger(presentation) {
  const card = createElement("article", "narrative-card clue-ledger-card");
  card.append(
    createElement("span", "card-label", "Strongest evidence"),
    createElement("h4", "", "Clues that shaped the decisions"),
  );
  const list = createElement("ol", "clue-ledger");
  presentation.top_clues.slice(0, MAXIMUM_VISIBLE_CLUES).forEach((clue) => {
    list.append(createClueItem(clue));
  });
  if (!list.childElementCount) {
    list.append(createElement("li", "clue-empty", "Boundary frames and measured motion continuity."));
  }
  card.append(list);
  return card;
}


function createClueItem(clue) {
  const item = createElement("li", "clue-ledger-item");
  const copy = createElement("span");
  copy.append(
    createElement("small", "", humanize(clue.category)),
    createElement("strong", "", clue.statement),
  );
  item.append(
    copy,
    createElement("span", "clue-confidence", `${Math.round(clue.confidence * 100)}%`),
  );
  return item;
}


function createDecisionPipeline(presentation) {
  const method = presentation.method;
  const section = createElement("section", "decision-pipeline");
  const heading = createElement("div");
  heading.append(
    createElement("span", "card-label", method?.label ?? "Public decision trace"),
    createElement("h4", "", "How the missing footage was reconstructed"),
    createElement(
      "p", "presentation-helper",
      method?.description ?? "A concise audit of observable inputs and validated outputs.",
    ),
  );
  const steps = createElement("ol", "decision-steps");
  (method?.steps ?? fallbackMethodSteps(presentation)).forEach((step, index) => {
    steps.append(createMethodStep(step, index));
  });
  section.append(heading, steps);
  return section;
}


function fallbackMethodSteps(presentation) {
  return [
    { title: "Observe", description: "Analyze only the visible source footage." },
    { title: "Measure", description: "Track entities and motion around each gap." },
    { title: "Decide", description: "Compare evidence-bounded motion hypotheses." },
    {
      title: "Patch",
      description: `Render and stitch ${presentation.gaps.length} inferred intervals.`,
    },
  ];
}


function createMethodStep(step, index) {
  const item = createElement("li", "decision-step");
  item.append(
    createElement("span", "decision-step-number", String(index + 1).padStart(2, "0")),
    createElement("strong", "", step.title),
    createElement("p", "", step.description),
  );
  return item;
}


function createMetric(value, label) {
  const metric = createElement("span", "case-metric");
  metric.append(
    createElement("strong", "", value),
    createElement("small", "", label),
  );
  return metric;
}


function createFact(value, label) {
  const fact = createElement("span", "evidence-fact");
  fact.append(
    createElement("strong", "", value),
    createElement("small", "", label),
  );
  return fact;
}


function formatDuration(totalSeconds) {
  const boundedSeconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(boundedSeconds / 60);
  const seconds = boundedSeconds % 60;
  return minutes ? `${minutes}m ${seconds}s` : `${seconds}s`;
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
