// @ts-check
// Records the live WebGL canvas to a downloadable WebM clip using the browser's own
// MediaRecorder over canvas.captureStream(). This is a capture of the interactive preview,
// clearly labelled as such — never presented as a recovered original. Fails gracefully
// (and reversibly) when the browser lacks MediaRecorder or a supported codec.

const PREFERRED_MIME_TYPES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
];

export class CaptureUnsupportedError extends Error {}

export class CaptureController {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} [frameRate]
   */
  constructor(canvas, frameRate = 30) {
    this._canvas = canvas;
    this._frameRate = frameRate;
    /** @type {MediaRecorder | null} */
    this._recorder = null;
    /** @type {Blob[]} */
    this._chunks = [];
  }

  /** @returns {boolean} */
  static isSupported() {
    return typeof window !== "undefined"
      && typeof window.MediaRecorder === "function"
      && PREFERRED_MIME_TYPES.some((type) => window.MediaRecorder.isTypeSupported(type));
  }

  /** @returns {boolean} */
  get isRecording() {
    return this._recorder !== null && this._recorder.state === "recording";
  }

  /**
   * Begin recording. Throws CaptureUnsupportedError when capture is unavailable, leaving
   * the viewer otherwise untouched.
   */
  start() {
    if (!CaptureController.isSupported()) {
      throw new CaptureUnsupportedError("This browser cannot record the 3D preview");
    }
    if (this.isRecording) return;
    const mimeType = PREFERRED_MIME_TYPES.find((type) => window.MediaRecorder.isTypeSupported(type));
    const stream = this._canvas.captureStream(this._frameRate);
    this._chunks = [];
    this._recorder = new MediaRecorder(stream, { mimeType });
    this._recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) this._chunks.push(event.data);
    });
    this._recorder.start();
  }

  /**
   * Stop recording and resolve with the recorded clip.
   * @returns {Promise<Blob>}
   */
  stop() {
    return new Promise((resolve, reject) => {
      if (!this._recorder) {
        reject(new CaptureUnsupportedError("No recording is in progress"));
        return;
      }
      const recorder = this._recorder;
      recorder.addEventListener("stop", () => {
        const blob = new Blob(this._chunks, { type: recorder.mimeType || "video/webm" });
        this._recorder = null;
        resolve(blob);
      }, { once: true });
      recorder.addEventListener("error", () => reject(new Error("Recording failed")), { once: true });
      recorder.stop();
    });
  }
}

/**
 * Trigger a browser download for a recorded clip. The caller owns the object URL lifetime.
 * @param {Blob} blob @param {string} filename
 */
export function downloadCapture(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
