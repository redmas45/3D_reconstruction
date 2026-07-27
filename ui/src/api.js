/**
 * Every call the interface makes. Kept in one place so the endpoint contract is
 * readable without opening components, and so a failure has one shape everywhere.
 */

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    // FastAPI reports failures as {detail}; anything else is shown verbatim rather
    // than swallowed, because a silent no-op is worse than an ugly message.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* not JSON; keep the status line */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

export const getHealth = () => request("/api/health");
export const getConfig = () => request("/api/config");
export const listJobs = () => request("/api/jobs");
export const getJob = (id) => request(`/api/jobs/${id}`);
export const cancelJob = (id) => request(`/api/jobs/${id}/cancel`, { method: "POST" });
export const deleteJob = (id) => request(`/api/jobs/${id}`, { method: "DELETE" });

export function uploadVideo(file, onProgress) {
  // XMLHttpRequest rather than fetch: upload progress is worth having for a file that
  // can be hundreds of megabytes, and fetch cannot report it.
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let detail = `Upload failed (${xhr.status})`;
        try {
          detail = JSON.parse(xhr.responseText).detail ?? detail;
        } catch {
          /* keep the status line */
        }
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => reject(new Error("The upload could not reach the backend."));
    xhr.send(form);
  });
}

/**
 * Subscribe to a job's live state. Returns an unsubscribe function.
 *
 * The stream carries the whole current state on every change rather than deltas, so a
 * reconnecting browser is immediately correct instead of having to replay what it
 * missed while it was away.
 */
export function streamJob(id, onUpdate, onDone) {
  const source = new EventSource(`/api/jobs/${id}/stream`);
  source.addEventListener("update", (event) => onUpdate(JSON.parse(event.data)));
  source.addEventListener("done", () => {
    source.close();
    if (onDone) onDone();
  });
  source.addEventListener("deleted", () => source.close());
  source.onerror = () => source.close();
  return () => source.close();
}

export const plateUrl = (id) => `/api/jobs/${id}/plate`;
export const videoUrl = (id) => `/api/jobs/${id}/video`;
export const gapVideoUrl = (id, gap) => `/api/jobs/${id}/gaps/${gap}/video`;
export const gapTruthUrl = (id, gap) => `/api/jobs/${id}/gaps/${gap}/truth`;
