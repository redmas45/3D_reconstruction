// @ts-check

const JOBS_ENDPOINT = "/api/jobs";
const NO_CACHE_REQUEST = Object.freeze({ cache: "no-store" });

/**
 * @typedef {Object} ProcessingJob
 * @property {string} id
 * @property {string} source_name
 * @property {"queued"|"processing"|"completed"|"failed"} status
 * @property {string} stage
 * @property {number} progress
 * @property {string} detail
 * @property {string} created_at
 * @property {string|null} completed_at
 * @property {number} elapsed_seconds
 * @property {number|null} eta_seconds
 * @property {string|null} error
 * @property {string|null} output_url
 * @property {string|null} download_url
 * @property {number|null} size_bytes
 * @property {boolean} is_legacy_output
 * @property {"blender"|"2d"} renderer_mode
 */

/** @returns {Promise<ProcessingJob[]>} */
export async function fetchProcessingJobs() {
  const response = await fetch(JOBS_ENDPOINT, NO_CACHE_REQUEST);
  if (!response.ok) throw new Error("The local server could not list processing jobs");
  const payload = await response.json();
  if (!payload || !Array.isArray(payload.jobs)) throw new Error("The server returned an invalid job list");
  return payload.jobs;
}

/**
 * @param {File} videoFile
 * @param {"blender"|"2d"} rendererMode
 * @param {(percentage: number) => void} reportUploadProgress
 * @returns {Promise<ProcessingJob>}
 */
export function uploadVideoJob(videoFile, rendererMode, reportUploadProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", JOBS_ENDPOINT);
    request.setRequestHeader("Content-Type", videoFile.type || "application/octet-stream");
    request.setRequestHeader("X-File-Name", encodeURIComponent(videoFile.name));
    request.setRequestHeader("X-Renderer-Mode", rendererMode);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      reportUploadProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.addEventListener("load", () => resolveUploadResponse(request, resolve, reject));
    request.addEventListener("error", () => reject(new Error("Could not reach the local processing server")));
    request.send(videoFile);
  });
}

/** @param {string} jobId @returns {Promise<void>} */
export async function deleteProcessingJob(jobId) {
  const response = await fetch(`${JOBS_ENDPOINT}/${jobId}`, { method: "DELETE" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Deletion failed");
}

/**
 * @param {XMLHttpRequest} request
 * @param {(job: ProcessingJob) => void} resolve
 * @param {(reason: Error) => void} reject
 */
function resolveUploadResponse(request, resolve, reject) {
  let payload;
  try {
    payload = JSON.parse(request.responseText);
  } catch (error) {
    reject(new Error("The server returned an invalid upload response", { cause: error }));
    return;
  }
  if (request.status < 200 || request.status >= 300) {
    reject(new Error(payload.error || "Upload failed"));
    return;
  }
  if (!payload.job || typeof payload.job.id !== "string") {
    reject(new Error("The server did not return a valid processing job"));
    return;
  }
  resolve(payload.job);
}
