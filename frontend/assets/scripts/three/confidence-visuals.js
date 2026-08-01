// @ts-check
// Turns confidence and uncertainty into honest visual language. Higher-confidence
// figures are solid; lower-confidence figures become translucent and carry a widening
// uncertainty corridor along their path, coloured green/amber/red. Colour is never the
// only signal — the dashboard renders matching text labels — but in the 3D view the
// corridor and opacity communicate how much to trust each reconstructed track.

import * as THREE from "../../vendor/three/three.module.js";
import { clamp } from "./scene-space.js";

const STRONG_SUPPORT = 0.75;
const MIXED_SUPPORT = 0.5;
const GREEN = new THREE.Color(0.20, 0.78, 0.45);
const AMBER = new THREE.Color(0.95, 0.68, 0.20);
const RED = new THREE.Color(0.90, 0.30, 0.30);

/**
 * @param {number} confidence
 * @returns {"strong" | "mixed" | "weak"}
 */
export function supportTier(confidence) {
  if (confidence >= STRONG_SUPPORT) return "strong";
  if (confidence >= MIXED_SUPPORT) return "mixed";
  return "weak";
}

/**
 * @param {number} confidence
 * @returns {THREE.Color}
 */
export function confidenceColor(confidence) {
  const tier = supportTier(confidence);
  if (tier === "strong") return GREEN.clone();
  if (tier === "mixed") return AMBER.clone();
  return RED.clone();
}

/**
 * Keep the actor readable. Confidence is communicated by the corridor, grounding disc,
 * and dashboard text; fading a person into the plate makes the result look broken.
 * @param {THREE.Object3D} model @param {number} confidence @param {string} fidelityTier
 */
export function applyConfidenceOpacity(model, confidence, fidelityTier) {
  void confidence;
  void fidelityTier;
  model.traverse((node) => {
    const mesh = /** @type {THREE.Mesh} */ (node);
    if (!mesh.material) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    materials.forEach((material) => {
      material.transparent = false;
      material.opacity = 1;
      material.depthWrite = true;
    });
  });
}

/**
 * A translucent corridor along the path whose half-width is the entity's positional
 * uncertainty. Wider + redder means less certain. Returns null when the path is a point.
 * @param {THREE.CatmullRomCurve3} curve
 * @param {number} radiusMetres @param {number} confidence
 * @returns {THREE.Mesh | null}
 */
export function buildUncertaintyCorridor(curve, radiusMetres, confidence) {
  const length = curve.getLength();
  if (length < 0.1) return null;
  const radius = clamp(radiusMetres || 0.4, 0.1, 3.0);
  const tubularSegments = Math.max(8, Math.round(length * 2));
  const geometry = new THREE.TubeGeometry(curve, tubularSegments, radius, 10, false);
  const material = new THREE.MeshBasicMaterial({
    color: confidenceColor(confidence),
    transparent: true,
    opacity: 0.14,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const corridor = new THREE.Mesh(geometry, material);
  corridor.name = "uncertainty-corridor";
  return corridor;
}

/**
 * A soft coloured disc under a figure's start position — a contact-shadow-like grounding
 * cue tinted by confidence.
 * @param {number} radius @param {number} confidence
 * @returns {THREE.Mesh}
 */
export function buildGroundingDisc(radius, confidence) {
  const geometry = new THREE.CircleGeometry(Math.max(0.2, radius), 20);
  geometry.rotateX(-Math.PI / 2);
  const material = new THREE.MeshBasicMaterial({
    color: confidenceColor(confidence),
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
  });
  const disc = new THREE.Mesh(geometry, material);
  disc.position.y = 0.01;
  return disc;
}
