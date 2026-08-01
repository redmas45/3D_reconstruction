// @ts-check
// Shared coordinate + math helpers for the reconstruction renderer.
//
// The backend manifest uses the domain world frame: X = right, Y = forward/depth,
// Z = up, in metres, ground at Z = 0. Three.js is Y-up. A single fixed rotation about
// the X axis (Z-up -> Y-up) maps one to the other while preserving right-handedness,
// so world geometry projects through the calibrated camera exactly as the backend
// intended. Every world point entering the scene goes through worldToScene().

import * as THREE from "../../vendor/three/three.module.js";

/**
 * Map a domain world point [x, y, z] (Z-up metres) into a Three.js Y-up vector.
 * (x, y, z) -> (x, z, -y): a -90 degree rotation about X. Ground (z=0) lands on y=0.
 * @param {number[]} world
 * @returns {THREE.Vector3}
 */
export function worldToScene(world) {
  const x = Number(world?.[0]) || 0;
  const y = Number(world?.[1]) || 0;
  const z = Number(world?.[2]) || 0;
  return new THREE.Vector3(x, z, -y);
}

/**
 * Heading in degrees, measured in the domain ground plane (0 = +Y forward, growing
 * toward +X), converted to a Three.js Y-axis rotation so a model's forward (-Z) faces
 * the travel direction.
 * @param {number} headingDegrees
 * @returns {number} radians
 */
export function headingToSceneYaw(headingDegrees) {
  return THREE.MathUtils.degToRad(Number(headingDegrees) || 0);
}

/**
 * Convert a horizontal field of view (what the calibration reports, for a 36mm sensor)
 * into the vertical field of view a Three.js PerspectiveCamera expects.
 * @param {number} horizontalFovDegrees
 * @param {number} aspectRatio width / height
 * @returns {number} vertical fov in degrees
 */
export function horizontalToVerticalFov(horizontalFovDegrees, aspectRatio) {
  const horizontalRadians = THREE.MathUtils.degToRad(clampFov(horizontalFovDegrees));
  const verticalRadians = 2 * Math.atan(Math.tan(horizontalRadians / 2) / Math.max(0.0001, aspectRatio));
  return THREE.MathUtils.radToDeg(verticalRadians);
}

/**
 * A deterministic pseudo-random generator seeded from an integer, so a track's
 * appearance jitter is identical every time it is drawn.
 * @param {number} seed
 * @returns {() => number} function returning floats in [0, 1)
 */
export function seededRandom(seed) {
  let state = (Number(seed) >>> 0) || 1;
  return function next() {
    // Mulberry32: small, fast, good enough for cosmetic variation.
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let result = Math.imul(state ^ (state >>> 15), 1 | state);
    result = (result + Math.imul(result ^ (result >>> 7), 61 | result)) ^ result;
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Clamp a value into an inclusive range.
 * @param {number} value @param {number} minimum @param {number} maximum
 */
export function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function clampFov(fovDegrees) {
  return clamp(Number(fovDegrees) || 58, 1, 179);
}

/**
 * A colour from a normalized [r, g, b] triple, with a safe grey fallback.
 * @param {number[]} rgb
 * @returns {THREE.Color}
 */
export function colorFromArray(rgb) {
  if (!Array.isArray(rgb) || rgb.length < 3) return new THREE.Color(0.5, 0.5, 0.5);
  return new THREE.Color(
    clamp(Number(rgb[0]) || 0, 0, 1),
    clamp(Number(rgb[1]) || 0, 0, 1),
    clamp(Number(rgb[2]) || 0, 0, 1),
  );
}
