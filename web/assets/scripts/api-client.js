// @ts-check

const JOBS_ENDPOINT = "/api/jobs";
const NO_CACHE_REQUEST = Object.freeze({ cache: "no-store" });
const UPLOAD_TIMEOUT_MILLISECONDS = 300_000;

/**
 * @typedef {Object} JobActivity
 * @property {string} timestamp
 * @property {string} stage
 * @property {string} detail
 * @property {number} progress
 */

/**
 * @typedef {Object} ReasoningDecision
 * @property {number} gap_index
 * @property {string} gap_summary
 * @property {string[]} evidence_references
 * @property {string[]} clue_ids
 * @property {number} confidence
 * @property {string[]} unknowns
 * @property {ReasoningEntityDecision[]} entities
 */

/**
 * @typedef {Object} ReasoningEntityDecision
 * @property {string} entity_id
 * @property {string} selected_hypothesis_id
 * @property {string} decision_summary
 * @property {{id: string, reason: string}[]} rejected_hypotheses
 * @property {number} confidence
 */

/**
 * @typedef {Object} ReasoningSummary
 * @property {string} status
 * @property {number} [schema_version]
 * @property {string} mode
 * @property {string|null} deployment
 * @property {string|null} warning
 * @property {string[]} scene_clues
 * @property {{id:string, scope:string, category:string, statement:string, confidence:number}[]} [clues]
 * @property {string} [headline]
 * @property {string} [whole_video_summary]
 * @property {{statement:string, clue_ids:string[], gap_indexes:number[]}[]} [story_points]
 * @property {{gap_index:number, before_observed:string, inside_inferred:string, after_observed:string, confidence:number, unknowns:string[]}[]} [gap_summaries]
 * @property {boolean} [causal_link_supported]
 * @property {number} [confidence]
 * @property {string[]} [unknowns]
 * @property {ReasoningDecision[]} decisions
 */

/**
 * @typedef {Object} PresentationGap
 * @property {number} gap_index
 * @property {number} start_seconds
 * @property {number} end_seconds
 * @property {number} duration_seconds
 * @property {number} confidence
 * @property {string} before_observed
 * @property {string} inside_inferred
 * @property {string} after_observed
 * @property {string[]} unknowns
 */

/**
 * @typedef {Object} PresentationManifest
 * @property {number} schema_version
 * @property {string} title
 * @property {string} disclosure
 * @property {{duration_seconds:number, observed_fraction:number}} source
 * @property {{headline:string, summary:string, confidence:number, causal_link_supported:boolean, points:string[]}} story
 * @property {{id:string, category:string, statement:string, confidence:number}[]} top_clues
 * @property {PresentationGap[]} gaps
 * @property {{mode:string, engine:string, target_fps:number, hybrid_static_backplate:boolean}} render
 */

/**
 * @typedef {Object} ProcessingJob
 * @property {string} id
 * @property {string} source_name
 * @property {"queued"|"processing"|"cancelling"|"cancelled"|"completed"|"failed"} status
 * @property {string} stage
 * @property {number} progress
 * @property {string} detail
 * @property {string} created_at
 * @property {string|null} completed_at
 * @property {number} elapsed_seconds
 * @property {number|null} eta_seconds
 * @property {"waiting"|"estimating"|"counting_down"|"recalibrating"|"finished"} eta_status
 * @property {JobActivity[]} activity_log
 * @property {string|null} error
 * @property {string|null} output_url
 * @property {string|null} download_url
 * @property {number|null} size_bytes
 * @property {boolean} is_legacy_output
 * @property {"blender"|"2d"} renderer_mode
 * @property {ReasoningSummary|null} reasoning
 * @property {PresentationManifest|null} presentation
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
    request.timeout = UPLOAD_TIMEOUT_MILLISECONDS;
    request.setRequestHeader("Content-Type", videoFile.type || "application/octet-stream");
    request.setRequestHeader("X-File-Name", encodeURIComponent(videoFile.name));
    request.setRequestHeader("X-Renderer-Mode", rendererMode);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      reportUploadProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.addEventListener("load", () => resolveUploadResponse(request, resolve, reject));
    request.addEventListener("error", () => reject(new Error("Could not reach the local processing server")));
    request.addEventListener("timeout", () => reject(new Error("Video upload timed out after 5 minutes")));
    request.send(videoFile);
  });
}

/** @param {string} jobId @returns {Promise<void>} */
export async function deleteProcessingJob(jobId) {
  const response = await fetch(`${JOBS_ENDPOINT}/${jobId}`, { method: "DELETE" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Deletion failed");
}

/** @param {string} jobId @returns {Promise<ProcessingJob>} */
export async function cancelProcessingJob(jobId) {
  const response = await fetch(`${JOBS_ENDPOINT}/${jobId}/cancel`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Cancellation failed");
  if (!payload.job || typeof payload.job.id !== "string") {
    throw new Error("The server did not return a valid cancelled job");
  }
  return payload.job;
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
