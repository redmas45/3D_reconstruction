// @ts-check

const PREFERRED_MIME_TYPES = [
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];

export class CompositeExportUnsupportedError extends Error {}

/**
 * Return whether this browser can capture a video and the transparent Three.js layer
 * into one reviewable file.
 * @param {HTMLVideoElement} video
 * @returns {boolean}
 */
export function compositeExportSupported(video) {
  return typeof window !== "undefined"
    && typeof window.MediaRecorder === "function"
    && typeof video.captureStream === "function"
    && typeof HTMLCanvasElement.prototype.captureStream === "function"
    && PREFERRED_MIME_TYPES.some((type) => window.MediaRecorder.isTypeSupported(type));
}

/**
 * Record the complete timeline while drawing the transparent Three.js layer over it.
 * This stays entirely in the browser and intentionally produces WebM; the original
 * evidence-safe MP4 remains available separately.
 *
 * @param {HTMLVideoElement} video
 * @param {HTMLElement} overlay
 * @param {number} frameRate
 * @param {string} filename
 * @param {() => void} syncOverlay
 * @returns {Promise<void>}
 */
export function exportCompositeTimeline(video, overlay, frameRate, filename, syncOverlay) {
  if (!compositeExportSupported(video)) {
    throw new CompositeExportUnsupportedError("This browser cannot export a composite video");
  }
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) throw new CompositeExportUnsupportedError("Video metadata is not ready");

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new CompositeExportUnsupportedError("Canvas export is unavailable");

  const outputStream = canvas.captureStream(Math.max(1, Math.round(frameRate || 30)));
  const sourceStream = video.captureStream();
  sourceStream.getAudioTracks().forEach((track) => outputStream.addTrack(track));
  const mimeType = PREFERRED_MIME_TYPES.find((type) => window.MediaRecorder.isTypeSupported(type));
  const recorder = new MediaRecorder(outputStream, { mimeType });
  const originalTime = video.currentTime;
  const wasPlaying = !video.paused && !video.ended;
  const chunks = [];
  let animationFrame = 0;
  let restored = false;

  return new Promise((resolve, reject) => {
    const restoreVideo = () => {
      if (restored) return;
      restored = true;
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      video.currentTime = originalTime;
      if (wasPlaying) void video.play();
      outputStream.getTracks().forEach((track) => track.stop());
    };
    const fail = (error) => {
      restoreVideo();
      reject(error instanceof Error ? error : new Error("Composite export failed"));
    };
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("error", () => fail(new Error("Composite recording failed")), { once: true });
    recorder.addEventListener("stop", () => {
      const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
      downloadBlob(blob, filename);
      restoreVideo();
      resolve();
    }, { once: true });

    const drawFrame = () => {
      syncOverlay();
      context.drawImage(video, 0, 0, width, height);
      const threeCanvas = overlay.querySelector("canvas");
      if (overlay.classList.contains("is-active") && threeCanvas) {
        context.drawImage(threeCanvas, 0, 0, width, height);
      }
      if (video.ended) {
        recorder.stop();
        return;
      }
      animationFrame = window.requestAnimationFrame(drawFrame);
    };

    video.pause();
    video.currentTime = 0;
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      video.removeEventListener("seeked", start);
      recorder.start(250);
      void video.play().then(drawFrame).catch(fail);
    };
    video.addEventListener("seeked", start, { once: true });
    if (video.currentTime === 0) window.setTimeout(start, 0);
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
}
