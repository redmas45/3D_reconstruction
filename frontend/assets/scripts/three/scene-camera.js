// @ts-check
// Builds the Three.js camera from the backend's calibrated camera contract, so what the
// viewer sees is the geometry the evidence supports rather than a flattering invention.
//
// The contract gives a world-space position and look-at (Z-up metres), a *horizontal*
// field of view (36mm-sensor convention), and a calibration confidence. We map the pose
// through the shared Z-up -> Y-up transform and convert the horizontal FOV to the
// vertical FOV a PerspectiveCamera wants, using the source frame's aspect ratio.

import * as THREE from "../../vendor/three/three.module.js";
import { worldToScene, horizontalToVerticalFov } from "./scene-space.js";

const NEAR_PLANE_METRES = 0.05;
const FAR_PLANE_METRES = 400;

/**
 * @param {object} cameraContract a manifest camera block
 * @param {number} sourceAspectRatio width / height of the source video
 * @returns {THREE.PerspectiveCamera}
 */
export function buildSceneCamera(cameraContract, sourceAspectRatio) {
  const contract = /** @type {any} */ (cameraContract || {});
  const verticalFov = horizontalToVerticalFov(
    contract.field_of_view_degrees ?? 58, sourceAspectRatio,
  );
  const camera = new THREE.PerspectiveCamera(
    verticalFov, sourceAspectRatio, NEAR_PLANE_METRES, FAR_PLANE_METRES,
  );
  applyCameraPose(camera, contract);
  return camera;
}

/**
 * Reposition an existing camera to a (possibly different, per-gap) camera contract
 * without reallocating it.
 * @param {THREE.PerspectiveCamera} camera
 * @param {object} cameraContract
 * @param {number} sourceAspectRatio
 */
export function applyCameraContract(camera, cameraContract, sourceAspectRatio) {
  const contract = /** @type {any} */ (cameraContract || {});
  camera.fov = horizontalToVerticalFov(contract.field_of_view_degrees ?? 58, sourceAspectRatio);
  camera.aspect = sourceAspectRatio;
  applyCameraPose(camera, contract);
  camera.updateProjectionMatrix();
}

/**
 * @param {THREE.PerspectiveCamera} camera @param {any} contract
 */
function applyCameraPose(camera, contract) {
  const position = worldToScene(contract.position || [0, 0, 1.6]);
  const lookAt = worldToScene(contract.look_at || [0, 10, 1.6]);
  camera.position.copy(position);
  camera.up.set(0, 1, 0);
  camera.lookAt(lookAt);
  camera.updateProjectionMatrix();
}
