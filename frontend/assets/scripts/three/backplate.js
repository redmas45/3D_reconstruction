// @ts-check
// The scene backdrop. The browser is never handed source frames (they may contain hidden
// footage), so the backplate is an honest stylized gradient rather than a photographic
// plate — a calm studio backdrop that reads as an intentional 3D stage. If the backend
// ever exports an explicitly-visible boundary frame as an allowed texture, this is the
// single place that would consume it; today it does not.

import * as THREE from "../../vendor/three/three.module.js";
import { colorFromArray } from "./scene-space.js";

/**
 * Build a vertical-gradient background texture derived from the environment palette.
 * @param {object} environmentContract
 * @returns {THREE.Texture}
 */
export function buildBackplateTexture(environmentContract) {
  const contract = /** @type {any} */ (environmentContract || {});
  const ground = colorFromArray(contract.ground_color || [0.05, 0.06, 0.08]);
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  if (context) {
    const gradient = context.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, tint(ground, 1.9));
    gradient.addColorStop(0.55, tint(ground, 1.25));
    gradient.addColorStop(1, tint(ground, 0.85));
    context.fillStyle = gradient;
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/**
 * @param {THREE.Color} color @param {number} factor
 */
function tint(color, factor) {
  const r = Math.round(Math.min(1, color.r * factor) * 255);
  const g = Math.round(Math.min(1, color.g * factor) * 255);
  const b = Math.round(Math.min(1, color.b * factor) * 255);
  return `rgb(${r}, ${g}, ${b})`;
}
