// @ts-check
// Detects whether this browser/GPU can run the WebGL renderer at all, and produces a
// clear, non-technical message when it cannot, so the scene degrades to an explanation
// rather than a blank canvas.

/**
 * @returns {boolean} true when a WebGL context can be created.
 */
export function webglIsAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(
      window.WebGLRenderingContext
      && (canvas.getContext("webgl2") || canvas.getContext("webgl")),
    );
  } catch (error) {
    return false;
  }
}

/**
 * @param {string} [reason]
 * @returns {HTMLElement}
 */
export function createWebglFallbackNotice(reason) {
  const notice = document.createElement("div");
  notice.className = "three-fallback-notice";
  notice.setAttribute("role", "status");
  const title = document.createElement("strong");
  title.textContent = "3D preview unavailable on this device";
  const detail = document.createElement("p");
  detail.textContent = reason
    || "This browser or GPU does not support WebGL, so the interactive 3D reconstruction "
    + "cannot render here. The evidence, decisions, and gap timeline remain fully available.";
  notice.append(title, detail);
  return notice;
}
