// @ts-check
// Deterministic locomotion. A path is a centripetal Catmull-Rom curve through the world
// waypoints (start boundary, inferred midpoint(s), end boundary) — never a straight
// start-to-end lerp. Position and facing come from the curve and its tangent; the walk
// cycle and wheel spin are driven by distance travelled, not by frame index, so speed and
// stride stay physically coupled and nothing teleports or jitters.

import * as THREE from "../../vendor/three/three.module.js";
import { worldToScene } from "./scene-space.js";

const MINIMUM_CURVE_SPAN_METRES = 0.02;
const HIP_SWING_RADIANS = 0.6;
const KNEE_BEND_RADIANS = 0.9;
const ARM_SWING_RADIANS = 0.5;
const TORSO_BOB_METRES = 0.03;
const IDLE_SWAY_RADIANS = 0.05;
const WALK_MOVING_SPEED = 0.2;
const MINIMUM_FOOT_GROUND_OFFSET = -0.18;
const MAXIMUM_FOOT_GROUND_OFFSET = 0.18;

/**
 * Build the scene-space path curve for an entity from its manifest waypoints.
 * @param {object} entity
 * @returns {{ curve: THREE.CatmullRomCurve3, length: number, points: THREE.Vector3[] }}
 */
export function buildPathCurve(entity) {
  const waypoints = /** @type {any[]} */ ((entity && /** @type {any} */ (entity).waypoints) || []);
  const points = waypoints.map((waypoint) => worldToScene(waypoint.world));
  const deduped = dedupe(points);
  if (deduped.length < 2) {
    // A held position: a tiny two-point curve keeps the API uniform without motion.
    const anchor = deduped[0] || new THREE.Vector3();
    const nudged = anchor.clone().add(new THREE.Vector3(0, 0, MINIMUM_CURVE_SPAN_METRES));
    const curve = new THREE.CatmullRomCurve3([anchor, nudged], false, "centripetal");
    return { curve, length: 0, points: deduped };
  }
  const curve = new THREE.CatmullRomCurve3(deduped, false, "centripetal");
  return { curve, length: curve.getLength(), points: deduped };
}

/**
 * Position + facing at normalized progress t along the path.
 * @param {THREE.CatmullRomCurve3} curve @param {number} t
 * @returns {{ position: THREE.Vector3, yaw: number }}
 */
export function pathPose(curve, t) {
  const clamped = Math.min(1, Math.max(0, t));
  const position = curve.getPoint(clamped);
  const tangent = curve.getTangent(clamped);
  // Face the tangent: a model whose forward is -Z reaches heading with this yaw.
  const yaw = Math.atan2(-tangent.x, -tangent.z);
  return { position, yaw };
}

/**
 * Drive an articulated actor's joints for a walk/idle cycle.
 * @param {any} joints from buildActor
 * @param {{ phase: number, moving: boolean, intensity: number }} state
 */
export function animateWalk(joints, state) {
  const { phase, moving, intensity } = state;
  if (!joints || !joints.hips) return;
  if (!joints.leftLeg) {
    // Simplified silhouette: a gentle sway only, never fake articulated detail.
    joints.hips.rotation.z = Math.sin(phase) * IDLE_SWAY_RADIANS * (moving ? 1 : 0.4);
    return;
  }
  const swing = moving ? intensity : 0.12;
  const wave = Math.sin(phase);
  const counterWave = Math.sin(phase + Math.PI);

  joints.leftLeg.hip.rotation.x = wave * HIP_SWING_RADIANS * swing;
  joints.rightLeg.hip.rotation.x = counterWave * HIP_SWING_RADIANS * swing;
  // Knees bend only as the leg swings back (never hyperextend forward).
  joints.leftLeg.knee.rotation.x = Math.max(0, -wave) * KNEE_BEND_RADIANS * swing;
  joints.rightLeg.knee.rotation.x = Math.max(0, -counterWave) * KNEE_BEND_RADIANS * swing;
  // Arms swing opposite the legs.
  joints.leftArm.shoulder.rotation.x = counterWave * ARM_SWING_RADIANS * swing;
  joints.rightArm.shoulder.rotation.x = wave * ARM_SWING_RADIANS * swing;
  joints.leftArm.elbow.rotation.x = Math.max(0, counterWave) * 0.4 * swing;
  joints.rightArm.elbow.rotation.x = Math.max(0, wave) * 0.4 * swing;
  // Subtle vertical bob at twice the stride frequency and a small torso counter-rotation.
  joints.hips.position.y = joints.hips.userData.baseY ?? joints.hips.position.y;
  if (joints.hips.userData.baseY === undefined) joints.hips.userData.baseY = joints.hips.position.y;
  joints.hips.position.y = joints.hips.userData.baseY - Math.abs(Math.sin(phase)) * TORSO_BOB_METRES * swing;
  if (joints.torso) joints.torso.rotation.y = wave * 0.08 * swing;
}

/**
 * Evaluate a real skinned locomotion clip at a deterministic phase. The actor root is
 * still positioned by the evidence path; only the joint pose comes from the clip, so
 * baked root translation cannot fight the camera-calibrated trajectory.
 * @param {{ mixer: THREE.AnimationMixer, clipDuration: number, strideLength: number, speed: number, phaseOffset: number, footBones?: THREE.Bone[], model?: THREE.Object3D }} animated
 * @param {number} distanceMetres
 * @param {number} elapsedSeconds
 */
export function updateHumanoidMotion(animated, distanceMetres, elapsedSeconds) {
  if (!animated?.mixer) return;
  const duration = Math.max(0.1, Number(animated.clipDuration) || 1);
  const moving = Number(animated.speed) >= WALK_MOVING_SPEED;
  const phaseOffset = positiveModulo(Number(animated.phaseOffset) || 0, 1);
  const phase = moving
    ? positiveModulo(
      distanceMetres / Math.max(0.4, Number(animated.strideLength) || 1) + phaseOffset,
      1,
    )
    : positiveModulo((Number(elapsedSeconds) || 0) / duration + phaseOffset, 1);
  animated.mixer.setTime(phase * duration);
  groundHumanoidFeet(animated);
}

/**
 * Keep the lowest planted foot on the calibrated ground plane. This is a small
 * contact correction, not a hidden-world solve: it prevents the imported clip's
 * pelvis bob from making a character visibly float or sink between frames.
 * @param {{ footBones?: THREE.Bone[], model?: THREE.Object3D }} animated
 */
function groundHumanoidFeet(animated) {
  if (!animated.model || !animated.footBones?.length) return;
  animated.model.updateMatrixWorld(true);
  const worldPosition = new THREE.Vector3();
  let lowestFootY = Number.POSITIVE_INFINITY;
  animated.footBones.forEach((bone) => {
    bone.getWorldPosition(worldPosition);
    lowestFootY = Math.min(lowestFootY, worldPosition.y);
  });
  if (!Number.isFinite(lowestFootY)) return;
  const correction = THREE.MathUtils.clamp(
    -lowestFootY,
    MINIMUM_FOOT_GROUND_OFFSET,
    MAXIMUM_FOOT_GROUND_OFFSET,
  );
  animated.model.position.y += correction;
}

/**
 * Roll a vehicle's wheels by the ground distance advanced this frame.
 * @param {THREE.Mesh[]} wheels @param {number} deltaMetres @param {number} wheelRadius
 */
export function animateWheels(wheels, deltaMetres, wheelRadius) {
  if (!Array.isArray(wheels) || wheelRadius <= 0) return;
  const spin = deltaMetres / wheelRadius;
  wheels.forEach((wheel) => { wheel.rotation.x += spin; });
}

/**
 * Cycle phase from distance travelled: stride length and speed stay coupled.
 * @param {number} distanceMetres @param {number} strideLength
 * @param {number} phaseOffset @param {number} cadenceScale
 * @returns {number} radians
 */
export function gaitPhase(distanceMetres, strideLength, phaseOffset, cadenceScale) {
  const stride = Math.max(0.2, strideLength);
  const cycles = (distanceMetres / stride) * (cadenceScale || 1) + (phaseOffset || 0);
  return cycles * Math.PI * 2;
}

/** @param {number} value @param {number} modulus @returns {number} */
function positiveModulo(value, modulus) {
  return ((value % modulus) + modulus) % modulus;
}

/**
 * @param {THREE.Vector3[]} points
 * @returns {THREE.Vector3[]}
 */
function dedupe(points) {
  const result = [];
  for (const point of points) {
    const previous = result[result.length - 1];
    if (!previous || previous.distanceTo(point) > MINIMUM_CURVE_SPAN_METRES) {
      result.push(point);
    }
  }
  return result;
}
