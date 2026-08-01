// @ts-check
// Builds a procedural low-poly human with a real joint hierarchy, so the motion system
// can articulate a walk rather than slide a billboard. Nested groups form the rig:
// rotating a hip group swings the whole leg; a knee group nested inside it bends the
// shin; the same for shoulders and elbows. Feet are planted on the ground (y = 0), the
// torso stays connected, and proportions come deterministically from the track's seed
// and body-proportion scales, so the same person looks the same in every gap.
//
// Low-confidence figures get a simplified silhouette instead of invented detail.

import * as THREE from "../../vendor/three/three.module.js";
import { colorFromArray, seededRandom, clamp } from "./scene-space.js";

const REFERENCE_HEIGHT_METRES = 1.75;
const SKIN_TONES = [
  [0.82, 0.66, 0.53], [0.70, 0.52, 0.40], [0.53, 0.38, 0.28], [0.90, 0.76, 0.64],
];
// Fractions of standing height for each landmark. Kept anatomically plausible so the
// figure reads as a person at the 60-300px it occupies on screen.
const HIP_FRACTION = 0.50;
const SHOULDER_FRACTION = 0.82;
const HEAD_CENTRE_FRACTION = 0.935;
const THIGH_FRACTION = 0.245;
const SHIN_FRACTION = 0.245;
const UPPER_ARM_FRACTION = 0.185;
const FOREARM_FRACTION = 0.175;

/**
 * @typedef {Object} ActorRig
 * @property {THREE.Group} group root; place at ground position and set rotation.y
 * @property {boolean} simplified
 * @property {number} height standing height in metres
 * @property {number} strideLength metres advanced per full gait cycle
 * @property {any} joints named pivot groups for the motion system
 */

/**
 * @param {object} entity a manifest entity
 * @returns {ActorRig}
 */
export function buildActor(entity) {
  const data = /** @type {any} */ (entity);
  const random = seededRandom(data.appearance_seed || 1);
  const proportions = data.body_proportions || {};
  const height = REFERENCE_HEIGHT_METRES
    * clamp(Number(proportions.height_scale) || 1, 0.8, 1.2)
    * heightForProxy(data.proxy);
  const simplified = data.visual_fidelity_tier === "weak";
  const materials = buildMaterials(data.appearance || {}, random);
  const group = new THREE.Group();
  group.name = `actor-${data.track_id}`;
  if (simplified) {
    const joints = buildSilhouette(group, height, materials);
    return { group, simplified, height, strideLength: 0.7 * height, joints };
  }
  const joints = buildArticulatedBody(group, height, proportions, materials);
  return { group, simplified, height, strideLength: 0.78 * height, joints };
}

/**
 * @param {THREE.Group} group @param {number} height @param {any} proportions @param {any} materials
 */
function buildArticulatedBody(group, height, proportions, materials) {
  const shoulderScale = clamp(Number(proportions.shoulder_scale) || 1, 0.8, 1.2);
  const limbScale = clamp(Number(proportions.limb_scale) || 1, 0.85, 1.15);
  const shoulderWidth = 0.19 * height * shoulderScale;
  const hipWidth = 0.12 * height;
  const hipHeight = HIP_FRACTION * height;

  const hips = new THREE.Group();
  hips.position.y = hipHeight;
  group.add(hips);

  const torsoHeight = (SHOULDER_FRACTION - HIP_FRACTION) * height;
  const torso = capsule(torsoHeight, 0.115 * height, materials.upper);
  torso.position.y = torsoHeight / 2;
  hips.add(torso);

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.085 * height, 16, 12), materials.skin,
  );
  head.castShadow = true;
  head.position.y = (HEAD_CENTRE_FRACTION - HIP_FRACTION) * height;
  const neck = new THREE.Group();
  neck.position.y = torsoHeight;
  neck.add(head);
  hips.add(neck);

  const thighLength = THIGH_FRACTION * height * limbScale;
  const shinLength = SHIN_FRACTION * height * limbScale;
  const leftLeg = buildLeg(hips, -hipWidth / 2, thighLength, shinLength, height, materials);
  const rightLeg = buildLeg(hips, hipWidth / 2, thighLength, shinLength, height, materials);

  const upperArmLength = UPPER_ARM_FRACTION * height * limbScale;
  const forearmLength = FOREARM_FRACTION * height * limbScale;
  const shoulderY = (SHOULDER_FRACTION - HIP_FRACTION) * height * 0.92;
  const leftArm = buildArm(hips, -shoulderWidth / 2, shoulderY, upperArmLength, forearmLength, height, materials);
  const rightArm = buildArm(hips, shoulderWidth / 2, shoulderY, upperArmLength, forearmLength, height, materials);

  return { hips, torso, neck, leftLeg, rightLeg, leftArm, rightArm };
}

/**
 * A downward-hanging leg: hip pivot -> thigh, knee pivot -> shin + foot.
 */
function buildLeg(parent, offsetX, thighLength, shinLength, height, materials) {
  const hip = new THREE.Group();
  hip.position.set(offsetX, 0, 0);
  parent.add(hip);
  hip.add(segment(thighLength, 0.055 * height, materials.lower));

  const knee = new THREE.Group();
  knee.position.y = -thighLength;
  hip.add(knee);
  knee.add(segment(shinLength, 0.045 * height, materials.lower));

  const foot = new THREE.Mesh(
    new THREE.BoxGeometry(0.07 * height, 0.045 * height, 0.16 * height), materials.foot,
  );
  foot.castShadow = true;
  foot.position.set(0, -shinLength - 0.02 * height, 0.045 * height);
  knee.add(foot);
  return { hip, knee };
}

/**
 * A downward-hanging arm: shoulder pivot -> upper arm, elbow pivot -> forearm + hand.
 */
function buildArm(parent, offsetX, shoulderY, upperArmLength, forearmLength, height, materials) {
  const shoulder = new THREE.Group();
  shoulder.position.set(offsetX, shoulderY, 0);
  parent.add(shoulder);
  shoulder.add(segment(upperArmLength, 0.042 * height, materials.upper));

  const elbow = new THREE.Group();
  elbow.position.y = -upperArmLength;
  shoulder.add(elbow);
  elbow.add(segment(forearmLength, 0.036 * height, materials.skin));

  const hand = new THREE.Mesh(
    new THREE.SphereGeometry(0.038 * height, 10, 8), materials.skin,
  );
  hand.castShadow = true;
  hand.position.y = -forearmLength;
  elbow.add(hand);
  return { shoulder, elbow };
}

/**
 * A capsule segment that hangs downward from a pivot at its top end.
 */
function segment(length, radius, material) {
  const mesh = capsule(length, radius, material);
  mesh.position.y = -length / 2;
  return mesh;
}

function capsule(length, radius, material) {
  const cylinderLength = Math.max(0.01, length - 2 * radius);
  const mesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, cylinderLength, 4, 10), material,
  );
  mesh.castShadow = true;
  return mesh;
}

/**
 * A readable low-poly person for low-confidence figures. It stays deliberately simple,
 * but separate head, torso, arms, and legs read as a person instead of a translucent blob.
 */
function buildSilhouette(group, height, materials) {
  const hips = new THREE.Group();
  hips.position.y = height * 0.5;
  group.add(hips);

  const body = capsule(height * 0.42, height * 0.11, materials.silhouette);
  body.position.y = height * 0.21;
  hips.add(body);

  const legLength = height * 0.46;
  [-1, 1].forEach((side) => {
    const leg = capsule(legLength, height * 0.045, materials.lower);
    leg.position.set(side * height * 0.055, -height * 0.27, 0);
    hips.add(leg);
  });

  const armLength = height * 0.32;
  [-1, 1].forEach((side) => {
    const arm = capsule(armLength, height * 0.038, materials.upper);
    arm.position.set(side * height * 0.16, height * 0.10, 0);
    arm.rotation.z = side * -0.16;
    hips.add(arm);
  });

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(height * 0.09, 12, 10), materials.skin,
  );
  head.castShadow = true;
  head.position.y = height * 0.49;
  hips.add(head);
  return { hips };
}

/**
 * @param {any} appearance @param {() => number} random
 */
function buildMaterials(appearance, random) {
  const skinTone = SKIN_TONES[Math.floor(random() * SKIN_TONES.length)];
  const roughness = 0.85;
  const upperColor = appearance.upper_color || [0.08, 0.28, 0.38];
  const lowerColor = appearance.lower_color || [0.10, 0.14, 0.20];
  return {
    upper: new THREE.MeshStandardMaterial({ color: colorFromArray(upperColor), roughness }),
    lower: new THREE.MeshStandardMaterial({ color: colorFromArray(lowerColor), roughness }),
    foot: new THREE.MeshStandardMaterial({ color: new THREE.Color(0.08, 0.08, 0.09), roughness }),
    skin: new THREE.MeshStandardMaterial({ color: colorFromArray(skinTone), roughness }),
    silhouette: new THREE.MeshStandardMaterial({
      color: colorFromArray(upperColor), roughness, flatShading: true,
    }),
  };
}

/**
 * Non-person renderable proxies still route here occasionally; scale to their catalog
 * height so a cyclist is not drawn at pedestrian height.
 * @param {any} proxy
 */
function heightForProxy(proxy) {
  const catalogHeight = Number(proxy?.dimensions_metres?.[2]);
  if (!Number.isFinite(catalogHeight) || catalogHeight <= 0) return 1;
  return catalogHeight / REFERENCE_HEIGHT_METRES;
}
