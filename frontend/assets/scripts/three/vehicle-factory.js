// @ts-check
// Builds a procedural low-poly vehicle from primitives: body, cabin, windows, four
// grounded wheels, and lights. The model's long axis is its travel direction (local -Z),
// so the motion system can yaw the whole group to the path heading and spin the wheels by
// distance travelled. Dimensions come from the class proxy catalogue, so a bus is a bus
// and a bicycle is a bicycle; colour is seeded by track id for cross-gap consistency.

import * as THREE from "../../vendor/three/three.module.js";
import { colorFromArray, seededRandom, clamp } from "./scene-space.js";

const DEFAULT_DIMENSIONS = { length: 4.3, width: 1.8, height: 1.45 };
const WHEEL_RADIUS_FRACTION = 0.32;
const CABIN_HEIGHT_FRACTION = 0.42;

/**
 * @typedef {Object} VehicleModel
 * @property {THREE.Group} group
 * @property {THREE.Mesh[]} wheels
 * @property {number} wheelRadius
 * @property {number} height
 */

/**
 * @param {object} entity a manifest entity
 * @returns {VehicleModel}
 */
export function buildVehicle(entity) {
  const data = /** @type {any} */ (entity);
  const random = seededRandom(data.appearance_seed || 1);
  const dimensions = readDimensions(data.proxy);
  const bodyColor = colorFromArray((data.appearance || {}).vehicle_color || [0.4, 0.42, 0.46]);
  const group = new THREE.Group();
  group.name = `vehicle-${data.track_id}`;

  const wheelRadius = dimensions.height * WHEEL_RADIUS_FRACTION;
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: bodyColor, roughness: 0.5, metalness: 0.35 });
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0.05, 0.08, 0.11), roughness: 0.15, metalness: 0.6,
  });

  group.add(buildBody(dimensions, wheelRadius, bodyMaterial));
  const cabin = buildCabin(dimensions, wheelRadius, bodyMaterial, glassMaterial);
  if (cabin) group.add(cabin);
  const wheels = buildWheels(dimensions, wheelRadius, random);
  wheels.forEach((wheel) => group.add(wheel));
  group.add(...buildLights(dimensions, wheelRadius));
  return { group, wheels, wheelRadius, height: dimensions.height };
}

/**
 * @param {any} proxy
 */
function readDimensions(proxy) {
  const dimensions = proxy?.dimensions_metres;
  if (!Array.isArray(dimensions) || dimensions.length < 3) return { ...DEFAULT_DIMENSIONS };
  return {
    length: clamp(Number(dimensions[0]) || DEFAULT_DIMENSIONS.length, 0.6, 20),
    width: clamp(Number(dimensions[1]) || DEFAULT_DIMENSIONS.width, 0.3, 4),
    height: clamp(Number(dimensions[2]) || DEFAULT_DIMENSIONS.height, 0.6, 5),
  };
}

function buildBody(dimensions, wheelRadius, material) {
  const bodyHeight = dimensions.height * (1 - CABIN_HEIGHT_FRACTION);
  const geometry = new THREE.BoxGeometry(dimensions.width, bodyHeight, dimensions.length);
  const body = new THREE.Mesh(geometry, material);
  body.castShadow = true;
  body.receiveShadow = true;
  body.position.y = wheelRadius + bodyHeight / 2;
  return body;
}

function buildCabin(dimensions, wheelRadius, bodyMaterial, glassMaterial) {
  // A cabin only reads on larger vehicles; a bicycle/motorcycle skips it.
  if (dimensions.height < 1.15 || dimensions.length < 2.4) return null;
  const cabinHeight = dimensions.height * CABIN_HEIGHT_FRACTION;
  const cabinLength = dimensions.length * 0.5;
  const cabinWidth = dimensions.width * 0.92;
  const cabin = new THREE.Group();
  const shell = new THREE.Mesh(
    new THREE.BoxGeometry(cabinWidth, cabinHeight, cabinLength), bodyMaterial,
  );
  shell.castShadow = true;
  cabin.add(shell);
  const glass = new THREE.Mesh(
    new THREE.BoxGeometry(cabinWidth * 1.005, cabinHeight * 0.7, cabinLength * 0.96), glassMaterial,
  );
  cabin.add(glass);
  cabin.position.y = wheelRadius + dimensions.height * (1 - CABIN_HEIGHT_FRACTION) + cabinHeight / 2;
  cabin.position.z = -dimensions.length * 0.05;
  return cabin;
}

/**
 * Four wheels at the corners; axles run along local X so they roll about X.
 * @param {any} dimensions @param {number} wheelRadius @param {() => number} random
 * @returns {THREE.Mesh[]}
 */
function buildWheels(dimensions, wheelRadius, random) {
  const wheelWidth = dimensions.width * 0.12;
  const geometry = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 16);
  geometry.rotateZ(Math.PI / 2); // lay the cylinder so its axis is X
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0.03, 0.03, 0.035), roughness: 0.85,
  });
  const halfLength = dimensions.length / 2 - wheelRadius;
  const halfWidth = dimensions.width / 2;
  const offsets = [
    [-halfWidth, halfLength], [halfWidth, halfLength],
    [-halfWidth, -halfLength], [halfWidth, -halfLength],
  ];
  return offsets.map(([x, z]) => {
    const wheel = new THREE.Mesh(geometry.clone(), material);
    wheel.castShadow = true;
    wheel.position.set(x, wheelRadius, z);
    return wheel;
  });
}

/**
 * @param {any} dimensions @param {number} wheelRadius
 * @returns {THREE.Object3D[]}
 */
function buildLights(dimensions, wheelRadius) {
  const headlightMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color(1, 0.95, 0.8), emissive: new THREE.Color(0.9, 0.85, 0.6), emissiveIntensity: 0.8,
  });
  const taillightMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0.6, 0.05, 0.05), emissive: new THREE.Color(0.7, 0.05, 0.05), emissiveIntensity: 0.7,
  });
  const geometry = new THREE.SphereGeometry(dimensions.height * 0.07, 8, 6);
  const y = wheelRadius + dimensions.height * 0.28;
  const x = dimensions.width * 0.32;
  const frontZ = -dimensions.length / 2 + 0.02;
  const rearZ = dimensions.length / 2 - 0.02;
  return [
    lightMesh(geometry, headlightMaterial, -x, y, frontZ),
    lightMesh(geometry, headlightMaterial, x, y, frontZ),
    lightMesh(geometry, taillightMaterial, -x, y, rearZ),
    lightMesh(geometry, taillightMaterial, x, y, rearZ),
  ];
}

function lightMesh(geometry, material, x, y, z) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x, y, z);
  return mesh;
}
