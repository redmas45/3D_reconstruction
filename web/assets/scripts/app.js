// @ts-check

import { deleteProcessingJob, fetchProcessingJobs, uploadVideoJob } from "./api-client.js";
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
  planning: "Planning scene",
  rendering: "Rendering",
  evaluating: "Evaluating",
  stitching: "Stitching",
  completed: "Completed",
  failed: "Failed",
};

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
  elements.uploadProgress.classList.remove("hidden");
  try {
    const rendererMode = elements.rendererMode.value === "2d" ? "2d" : "blender";
    await uploadVideoJob(selectedVideo, rendererMode, (percentage) => {
      elements.uploadPercentage.textContent = `${percentage}%`;
      elements.uploadBar.style.width = `${percentage}%`;
    });
    showToast("Video queued for reconstruction.");
    clearSelectedFile();
    await loadJobs();
  } catch (error) {
    showToast(errorMessage(error), true);
    elements.startButton.disabled = false;
  } finally {
    elements.uploadProgress.classList.add("hidden");
  }
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
  const activeJobs = jobs.filter((job) => job.status !== "completed");
  elements.jobList.replaceChildren(...activeJobs.map(createJobCard));
  elements.jobsEmpty.classList.toggle("hidden", activeJobs.length > 0);

  const outputs = jobs.filter((job) => job.status === "completed" && job.output_url);
  elements.outputGrid.replaceChildren(...outputs.map(createOutputCard));
  elements.outputsEmpty.classList.toggle("hidden", outputs.length > 0);
  elements.outputCount.textContent = `${outputs.length} ${outputs.length === 1 ? "output" : "outputs"}`;
}

/** @param {import("./api-client.js").ProcessingJob} job @returns {HTMLElement} */
function createJobCard(job) {
  const card = createElement("article", `job-card ${job.status === "failed" ? "failed" : ""}`);
  card.append(createElement("span", "job-spinner"));
  const main = createElement("div", "job-main");
  const titleRow = createElement("div", "job-title-row");
  titleRow.append(createElement("strong", "", job.source_name));
  titleRow.append(createElement("span", "stage-badge", STAGE_LABELS[job.stage] || job.stage));
  main.append(titleRow);
  main.append(createElement("span", "renderer-badge", job.renderer_mode === "blender" ? "Blender 3D" : "2.5D fallback"));
  main.append(createElement("p", `job-detail ${job.error ? "job-error" : ""}`, job.error || job.detail));
  const track = createElement("div", "progress-track");
  const bar = createElement("span");
  bar.style.width = `${Math.round(job.progress * 100)}%`;
  track.append(bar);
  main.append(track);
  const times = createElement("div", "job-times");
  times.append(createElement("span", "", `Progress ${Math.round(job.progress * 100)}%`));
  times.append(createElement("span", "", `Elapsed ${formatDuration(job.elapsed_seconds)}`));
  if (job.status !== "failed") times.append(createElement("span", "", `ETA ${formatDuration(job.eta_seconds)}`));
  main.append(times);
  card.append(main);
  if (job.status === "failed") {
    const remove = createElement("button", "small-delete", "Remove");
    remove.type = "button";
    remove.addEventListener("click", () => openDeleteModal(job.id));
    card.append(remove);
  }
  return card;
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

async function loadJobs() {
  try {
    const jobs = await fetchProcessingJobs();
    elements.systemStatus.classList.remove("offline");
    renderJobs(jobs);
  } catch (error) {
    elements.systemStatus.classList.add("offline");
  }
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
loadJobs();
setInterval(loadJobs, JOB_REFRESH_INTERVAL_MILLISECONDS);
