// @ts-check

import { cancelProcessingJob, deleteProcessingJob, fetchProcessingJobs, uploadVideoJob } from "./api-client.js";
import { errorMessage, extractFileExtension, formatByteCount, formatDuration } from "./formatters.js";

const SUPPORTED_EXTENSIONS = new Set(["mp4", "mov", "avi", "mkv", "m4v", "webm", "mpeg", "mpg", "wmv"]);
const THEME_STORAGE_KEY = "reconstruct-theme";
const JOB_REFRESH_INTERVAL_MILLISECONDS = 1200;
const TOAST_VISIBILITY_MILLISECONDS = 4200;
const STAGE_LABELS = {
  queued: "Queued",
  validating: "Validating",
  selecting_gaps: "Selecting gaps",
  preparing: "Preparing evidence",
  detecting: "Detecting & tracking",
  extracting_clues: "Extracting clues",
  reasoning: "Reasoning from evidence",
  validating_decisions: "Validating decisions",
  planning: "Planning scene",
  rendering: "Rendering",
  evaluating: "Evaluating",
  stitching: "Stitching",
  cancelling: "Cancelling",
  cancelled: "Cancelled",
  completed: "Completed",
  failed: "Failed",
};
const PIPELINE_STEPS = Object.freeze([
  { stage: "validating", label: "Validate input", start: 0, end: 0.04 },
  { stage: "selecting_gaps", label: "Select evidence gaps", start: 0.04, end: 0.06 },
  { stage: "preparing", label: "Prepare boundary evidence", start: 0.06, end: 0.13 },
  { stage: "detecting", label: "Detect and track entities", start: 0.13, end: 0.49 },
  { stage: "planning", label: "Build bounded hypotheses", start: 0.49, end: 0.51 },
  { stage: "extracting_clues", label: "Write visible-only clue ledger", start: 0.51, end: 0.53 },
  { stage: "reasoning", label: "Select evidence-grounded hypotheses", start: 0.53, end: 0.55 },
  { stage: "validating_decisions", label: "Validate decision trace", start: 0.55, end: 0.58 },
  { stage: "rendering", label: "Render inferred gaps", start: 0.58, end: 0.85 },
  { stage: "evaluating", label: "Evaluate inferred gaps", start: 0.85, end: 0.94 },
  { stage: "stitching", label: "Stitch video and audio", start: 0.94, end: 1 },
]);

const elements = {
  videoInput: /** @type {HTMLInputElement} */ (requiredElement("#video-input")),
  dropZone: requiredElement("#drop-zone"),
  selectedFile: requiredElement("#selected-file"),
  fileType: requiredElement("#file-type"),
  fileName: requiredElement("#file-name"),
  fileSize: requiredElement("#file-size"),
  clearFile: /** @type {HTMLButtonElement} */ (requiredElement("#clear-file")),
  startButton: /** @type {HTMLButtonElement} */ (requiredElement("#start-button")),
  rendererMode: /** @type {HTMLSelectElement} */ (requiredElement("#renderer-mode")),
  uploadProgress: requiredElement("#upload-progress"),
  uploadPercentage: requiredElement("#upload-percentage"),
  uploadBar: /** @type {HTMLElement} */ (requiredElement("#upload-bar")),
  jobList: requiredElement("#job-list"),
  jobsEmpty: requiredElement("#jobs-empty"),
  outputGrid: requiredElement("#output-grid"),
  outputsEmpty: requiredElement("#outputs-empty"),
  outputCount: requiredElement("#output-count"),
  systemStatus: requiredElement("#system-status"),
  themeToggle: /** @type {HTMLButtonElement} */ (requiredElement("#theme-toggle")),
  deleteModal: requiredElement("#delete-modal"),
  cancelDelete: /** @type {HTMLButtonElement} */ (requiredElement("#cancel-delete")),
  confirmDelete: /** @type {HTMLButtonElement} */ (requiredElement("#confirm-delete")),
  toast: requiredElement("#toast"),
};

/** @type {File|null} */
let selectedVideo = null;
/** @type {string|null} */
let deleteJobId = null;
/** @type {number|null} */
let toastTimer = null;
/** @type {Set<string>} */
const expandedJobLogs = new Set();
/** @type {Map<string, {outputUrl: string, card: HTMLElement}>} */
const outputCardsByJobId = new Map();
let jobsRequestInFlight = false;
let followUpJobsRefreshNeeded = false;

/** @param {string} selector @returns {Element} */
function requiredElement(selector) {
  const element = document.querySelector(selector);
  if (element == null) throw new Error(`Required interface element is missing: ${selector}`);
  return element;
}

/** @returns {"dark"|"light"} */
function preferredTheme() {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
  if (storedTheme === "dark" || storedTheme === "light") return storedTheme;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

/** @param {"dark"|"light"} theme @param {boolean} persist @returns {void} */
function applyTheme(theme, persist) {
  document.documentElement.dataset.theme = theme;
  const nextTheme = theme === "dark" ? "light" : "dark";
  const accessibleLabel = `Switch to ${nextTheme} theme`;
  elements.themeToggle.setAttribute("aria-label", accessibleLabel);
  elements.themeToggle.title = accessibleLabel;
  if (persist) window.localStorage.setItem(THEME_STORAGE_KEY, theme);
}

/** @returns {void} */
function toggleTheme() {
  const currentTheme = document.documentElement.dataset.theme === "light" ? "light" : "dark";
  applyTheme(currentTheme === "dark" ? "light" : "dark", true);
}

/** @param {string} message @param {boolean} [isError] @returns {void} */
function showToast(message, isError = false) {
  if (toastTimer != null) window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("visible");
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), TOAST_VISIBILITY_MILLISECONDS);
}

/** @param {File} file @returns {void} */
function setSelectedFile(file) {
  const extension = extractFileExtension(file.name);
  if (!SUPPORTED_EXTENSIONS.has(extension)) {
    clearSelectedFile();
    showToast("Choose a supported video: MP4, MOV, AVI, MKV, M4V, WebM, MPEG, MPG or WMV.", true);
    return;
  }
  selectedVideo = file;
  elements.fileType.textContent = extension.toUpperCase();
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = formatByteCount(file.size);
  elements.selectedFile.classList.remove("hidden");
  elements.startButton.disabled = false;
}

/** @returns {void} */
function clearSelectedFile() {
  selectedVideo = null;
  elements.videoInput.value = "";
  elements.selectedFile.classList.add("hidden");
  elements.startButton.disabled = true;
}

/** @returns {Promise<void>} */
async function uploadVideo() {
  if (!selectedVideo) return;
  elements.startButton.disabled = true;
  resetUploadProgress();
  elements.uploadProgress.classList.remove("hidden");
  try {
    const rendererMode = elements.rendererMode.value === "2d" ? "2d" : "blender";
    const job = await uploadVideoJob(selectedVideo, rendererMode, (percentage) => {
      elements.uploadPercentage.textContent = `${percentage}%`;
      elements.uploadBar.style.width = `${percentage}%`;
    });
    expandedJobLogs.add(job.id);
    showToast("Video queued for reconstruction.");
    clearSelectedFile();
    await loadJobs();
  } catch (error) {
    showToast(errorMessage(error), true);
    elements.startButton.disabled = false;
  } finally {
    elements.uploadProgress.classList.add("hidden");
    resetUploadProgress();
  }
}

/** @returns {void} */
function resetUploadProgress() {
  elements.uploadPercentage.textContent = "0%";
  elements.uploadBar.style.width = "0%";
}

/** @param {string} tag @param {string} [className] @param {string|null} [text] @returns {HTMLElement} */
function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}

/** @param {import("./api-client.js").ProcessingJob[]} jobs @returns {void} */
function renderJobs(jobs) {
  const activeJobs = jobs.filter((job) => job.status !== "completed" || !job.output_url);
  elements.jobList.replaceChildren(...activeJobs.map(createJobCard));
  elements.jobsEmpty.classList.toggle("hidden", activeJobs.length > 0);

  const outputs = jobs.filter((job) => job.status === "completed" && job.output_url);
  renderOutputCards(outputs);
  elements.outputsEmpty.classList.toggle("hidden", outputs.length > 0);
  elements.outputCount.textContent = `${outputs.length} ${outputs.length === 1 ? "output" : "outputs"}`;
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLElement} */
function createJobCard(job) {
  const outputIsMissing = job.status === "completed" && !job.output_url;
  const terminalClass = outputIsMissing ? "failed" : (["failed", "cancelled"].includes(job.status) ? job.status : "");
  const card = createElement("article", `job-card ${terminalClass}`);
  card.append(createElement("span", "job-spinner"));
  const main = createElement("div", "job-main");
  const titleRow = createElement("div", "job-title-row");
  titleRow.append(createElement("strong", "", job.source_name));
  const stageLabel = outputIsMissing ? "Output unavailable" : (STAGE_LABELS[job.stage] || job.stage);
  titleRow.append(createElement("span", "stage-badge", stageLabel));
  main.append(titleRow);
  main.append(createElement("span", "renderer-badge", job.renderer_mode === "blender" ? "Blender 3D" : "2.5D fallback"));
  const detail = outputIsMissing
    ? "Processing completed, but the output file is unavailable. Remove this record and run it again."
    : (job.error || job.detail);
  main.append(createElement("p", `job-detail ${job.error || outputIsMissing ? "job-error" : ""}`, detail));
  const track = createElement("div", "progress-track");
  const bar = createElement("span");
  bar.style.width = `${Math.round(job.progress * 100)}%`;
  track.append(bar);
  main.append(track);
  const times = createElement("div", "job-times");
  times.append(createElement("span", "", `Progress ${Math.round(job.progress * 100)}%`));
  times.append(createElement("span", "", `Elapsed ${formatDuration(job.elapsed_seconds)}`));
  if (["queued", "processing"].includes(job.status)) {
    times.append(createElement("span", "eta-value", etaLabel(job)));
  }
  main.append(times);
  main.append(createActivityPanel(job));
  if (job.reasoning) main.append(createReasoningPanel(job.reasoning));
  card.append(main);
  const action = createJobAction(job);
  if (action) card.append(action);
  return card;
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {string} */
function etaLabel(job) {
  if (job.eta_status === "waiting") return "ETA waiting for worker";
  if (job.eta_status === "recalibrating") return "ETA recalculating after current task";
  if (job.eta_seconds == null) return "ETA estimating from live progress";
  return `ETA ~${formatDuration(job.eta_seconds)}`;
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLElement} */
function createActivityPanel(job) {
  const details = /** @type {HTMLDetailsElement} */ (createElement("details", "job-activity"));
  details.open = expandedJobLogs.has(job.id);
  const summary = createElement("summary");
  summary.append(createElement("span", "", "Live reconstruction activity"));
  summary.append(createElement("span", "activity-summary-note", "Current, completed and pending"));
  details.append(summary, createPipelineSteps(job), createActivityFeed(job));
  details.addEventListener("toggle", () => rememberActivityState(job.id, details.open));
  return details;
}

/** @param {import("./api-client.js").ReasoningSummary} reasoning @returns {HTMLElement} */
function createReasoningPanel(reasoning) {
  const details = /** @type {HTMLDetailsElement} */ (createElement("details", "reasoning-panel"));
  const summary = createElement("summary");
  summary.append(createElement("span", "", "Evidence decision trace"));
  summary.append(createElement("span", "reasoning-mode", reasoningModeLabel(reasoning)));
  details.append(summary);
  if (reasoning.warning) details.append(createElement("p", "reasoning-warning", reasoning.warning));
  const clues = createElement("ul", "reasoning-clues");
  reasoning.scene_clues.forEach((clue) => clues.append(createElement("li", "", clue)));
  details.append(clues);
  const decisions = createElement("div", "reasoning-decisions");
  reasoning.decisions.forEach((decision) => decisions.append(createReasoningDecision(decision)));
  details.append(decisions);
  return details;
}

/** @param {import("./api-client.js").ReasoningSummary} reasoning @returns {string} */
function reasoningModeLabel(reasoning) {
  if (reasoning.mode === "azure" || reasoning.mode === "azure_cache") {
    return `Azure ${reasoning.deployment || "reasoning"}`;
  }
  return "Deterministic fallback";
}

/** @param {import("./api-client.js").ReasoningDecision} decision @returns {HTMLElement} */
function createReasoningDecision(decision) {
  const card = createElement("article", "reasoning-decision");
  const heading = createElement("div", "reasoning-decision-heading");
  heading.append(createElement("strong", "", `Gap ${decision.gap_index + 1}: ${humanizeIdentifier(decision.selected_hypothesis_id)}`));
  heading.append(createElement("span", "", `${Math.round(decision.confidence * 100)}% confidence`));
  card.append(heading, createElement("p", "", decision.decision_summary));
  const evidence = decision.evidence_references.length
    ? decision.evidence_references.join(" · ")
    : "No specific evidence reference supplied";
  card.append(createElement("small", "reasoning-evidence", `Evidence: ${evidence}`));
  if (decision.rejected_hypotheses.length) {
    const rejected = decision.rejected_hypotheses
      .map((item) => `${humanizeIdentifier(item.id)} — ${item.reason}`)
      .join(" · ");
    card.append(createElement("small", "reasoning-rejected", `Rejected: ${rejected}`));
  }
  if (decision.unknowns.length) {
    card.append(createElement("small", "reasoning-unknowns", `Unknowns: ${decision.unknowns.join(" · ")}`));
  }
  return card;
}

/** @param {string} identifier @returns {string} */
function humanizeIdentifier(identifier) {
  return identifier.replaceAll("_", " ");
}

/** @param {string} jobId @param {boolean} isOpen @returns {void} */
function rememberActivityState(jobId, isOpen) {
  if (isOpen) expandedJobLogs.add(jobId);
  else expandedJobLogs.delete(jobId);
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLElement} */
function createPipelineSteps(job) {
  const steps = createElement("div", "pipeline-steps");
  for (const definition of PIPELINE_STEPS) {
    const state = pipelineStepState(definition, job);
    const row = createElement("div", `pipeline-step ${state}`);
    row.append(createElement("span", "", definition.label));
    row.append(createElement("span", "pipeline-step-state", pipelineStepLabel(definition, job, state)));
    steps.append(row);
  }
  return steps;
}

/**
 * @param {{stage: string, start: number, end: number}} definition
 * @param {import("./api-client.js").ProcessingJob} job
 * @returns {"completed"|"active"|"pending"|"failed"|"stopped"}
 */
function pipelineStepState(definition, job) {
  const terminalState = job.status === "failed" ? "failed" : (job.status === "cancelled" ? "stopped" : null);
  if (terminalState != null) {
    if (job.stage === definition.stage) return terminalState;
    if (job.progress >= definition.end) return "completed";
    if (job.progress >= definition.start) return terminalState;
    return "pending";
  }
  if (job.stage === definition.stage) return "active";
  if (job.progress >= definition.end) return "completed";
  if (job.progress > definition.start) return "active";
  return "pending";
}

/**
 * @param {{start: number, end: number}} definition
 * @param {import("./api-client.js").ProcessingJob} job
 * @param {"completed"|"active"|"pending"|"failed"|"stopped"} state
 * @returns {string}
 */
function pipelineStepLabel(definition, job, state) {
  if (state === "completed") return "Done";
  if (state === "pending") return "Pending";
  if (state === "failed") return "Failed";
  if (state === "stopped") return "Stopped";
  const stageRange = definition.end - definition.start;
  const localProgress = Math.min(1, Math.max(0, (job.progress - definition.start) / stageRange));
  return `${Math.round(localProgress * 100)}%`;
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLElement} */
function createActivityFeed(job) {
  const feed = createElement("ol", "activity-feed");
  const recentActivity = job.activity_log.slice(-8).reverse();
  for (const activity of recentActivity) {
    const item = createElement("li", "activity-item");
    item.append(createElement("time", "activity-time", activityTime(activity.timestamp)));
    item.append(createElement("span", "", activity.detail));
    item.append(createElement("span", "activity-progress", `${Math.round(activity.progress * 100)}%`));
    feed.append(item);
  }
  if (!recentActivity.length) feed.append(createElement("li", "activity-item", "Waiting for the first pipeline update…"));
  return feed;
}

/** @param {string} timestamp @returns {string} */
function activityTime(timestamp) {
  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.valueOf()) ? "Now" : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLButtonElement|null} */
function createJobAction(job) {
  if (["queued", "processing"].includes(job.status)) {
    const cancel = /** @type {HTMLButtonElement} */ (createElement("button", "small-cancel", "Cancel job"));
    cancel.type = "button";
    cancel.addEventListener("click", () => cancelActiveJob(job.id, cancel));
    return cancel;
  }
  if (job.status === "cancelling") {
    const cancelling = /** @type {HTMLButtonElement} */ (createElement("button", "small-cancel", "Cancelling…"));
    cancelling.type = "button";
    cancelling.disabled = true;
    return cancelling;
  }
  const removableWithoutOutput = job.status === "completed" && !job.output_url;
  if (!["failed", "cancelled"].includes(job.status) && !removableWithoutOutput) return null;
  const remove = /** @type {HTMLButtonElement} */ (createElement("button", "small-delete", "Remove"));
  remove.type = "button";
  remove.addEventListener("click", () => openDeleteModal(job.id));
  return remove;
}

/** @param {string} jobId @param {HTMLButtonElement} button @returns {Promise<void>} */
async function cancelActiveJob(jobId, button) {
  button.disabled = true;
  button.textContent = "Cancelling…";
  try {
    await cancelProcessingJob(jobId);
    showToast("Cancellation requested. Active render processes are stopping.");
    await loadJobs();
  } catch (error) {
    showToast(errorMessage(error), true);
    button.disabled = false;
    button.textContent = "Cancel job";
  }
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLElement} */
function createOutputCard(job) {
  const card = createElement("article", "output-card");
  const frame = createElement("div", "video-frame");
  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  video.src = job.output_url;
  frame.append(video);
  card.append(frame);
  const body = createElement("div", "output-body");
  body.append(createElement("h3", "", job.source_name.replace(/(\.[^.]+)$/, "") + " — reconstructed"));
  const completionDate = job.completed_at ? new Date(job.completed_at).toLocaleString() : "Completed";
  body.append(createElement("p", "output-meta", `${formatByteCount(job.size_bytes)} · ${formatDuration(job.elapsed_seconds)} processing · ${completionDate}`));
  if (job.reasoning) body.append(createReasoningPanel(job.reasoning));
  const actions = createElement("div", "output-actions");
  const download = createElement("a", "action-button", "Download video");
  download.href = job.download_url;
  const remove = createElement("button", "action-button delete-button", "×");
  remove.type = "button";
  remove.setAttribute("aria-label", `Delete ${job.source_name}`);
  remove.addEventListener("click", () => openDeleteModal(job.id));
  actions.append(download, remove);
  body.append(actions);
  card.append(body);
  return card;
}

/** @param {import("./api-client.js").ProcessingJob[]} jobs @returns {void} */
function renderOutputCards(jobs) {
  const visibleJobIds = new Set(jobs.map((job) => job.id));
  for (const [jobId, cached] of outputCardsByJobId) {
    if (visibleJobIds.has(jobId)) continue;
    cached.card.remove();
    outputCardsByJobId.delete(jobId);
  }
  jobs.forEach((job, index) => placeOutputCard(job, index));
}

/** @param {import("./api-client.js").ProcessingJob} job @param {number} index @returns {void} */
function placeOutputCard(job, index) {
  const outputUrl = job.output_url || "";
  let cached = outputCardsByJobId.get(job.id);
  if (cached && cached.outputUrl !== outputUrl) {
    cached.card.remove();
    outputCardsByJobId.delete(job.id);
    cached = undefined;
  }
  const card = cached?.card || createOutputCard(job);
  if (!cached) outputCardsByJobId.set(job.id, { outputUrl, card });
  const cardAtIndex = elements.outputGrid.children.item(index);
  if (cardAtIndex !== card) elements.outputGrid.insertBefore(card, cardAtIndex);
}

async function loadJobs() {
  if (jobsRequestInFlight) {
    followUpJobsRefreshNeeded = true;
    return;
  }
  jobsRequestInFlight = true;
  try {
    const jobs = await fetchProcessingJobs();
    elements.systemStatus.classList.remove("offline");
    renderJobs(jobs);
  } catch (error) {
    elements.systemStatus.classList.add("offline");
  } finally {
    jobsRequestInFlight = false;
    if (followUpJobsRefreshNeeded) {
      followUpJobsRefreshNeeded = false;
      void loadJobs();
    }
  }
}

/** @returns {Promise<void>} */
async function pollJobs() {
  await loadJobs();
  window.setTimeout(pollJobs, JOB_REFRESH_INTERVAL_MILLISECONDS);
}

/** @param {string} jobId @returns {void} */
function openDeleteModal(jobId) {
  deleteJobId = jobId;
  elements.deleteModal.classList.remove("hidden");
  elements.cancelDelete.focus();
}

function closeDeleteModal() {
  deleteJobId = null;
  elements.deleteModal.classList.add("hidden");
}

async function deleteSelectedJob() {
  if (!deleteJobId) return;
  elements.confirmDelete.disabled = true;
  try {
    await deleteProcessingJob(deleteJobId);
    closeDeleteModal();
    showToast("Reconstruction and all job-owned files were deleted.");
    await loadJobs();
  } catch (error) {
    showToast(errorMessage(error), true);
  } finally {
    elements.confirmDelete.disabled = false;
  }
}

elements.videoInput.addEventListener("change", () => {
  const selectedFiles = elements.videoInput.files;
  if (selectedFiles?.length) setSelectedFile(selectedFiles[0]);
});
elements.clearFile.addEventListener("click", clearSelectedFile);
elements.startButton.addEventListener("click", uploadVideo);
elements.themeToggle.addEventListener("click", toggleTheme);
elements.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.dropZone.classList.add("dragging");
});
elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("dragging"));
elements.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  if (event.dataTransfer?.files.length) setSelectedFile(event.dataTransfer.files[0]);
});
elements.cancelDelete.addEventListener("click", closeDeleteModal);
elements.confirmDelete.addEventListener("click", deleteSelectedJob);
elements.deleteModal.addEventListener("click", (event) => {
  if (event.target === elements.deleteModal) closeDeleteModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.deleteModal.classList.contains("hidden")) closeDeleteModal();
});

applyTheme(preferredTheme(), false);
void pollJobs();
