// @ts-check
// Builds the stylized stage the reconstructed figures stand on: a grounded plane at the
// world's Z=0, soft studio lighting, and an optional reference grid for debug mode. The
// look is deliberately clean and low-poly rather than photoreal, and it is honest — no
// source footage is textured in, because the browser is never given hidden frames.

import * as THREE from "../../vendor/three/three.module.js";
import { colorFromArray } from "./scene-space.js";

const GROUND_SIZE_METRES = 200;
const GRID_DIVISIONS = 100;
const KEY_LIGHT_INTENSITY = 2.1;
const FILL_LIGHT_INTENSITY = 0.55;
const AMBIENT_INTENSITY = 0.65;

/**
 * @param {object} environmentContract a manifest gap.environment block
 * @returns {{ group: THREE.Group, grid: THREE.GridHelper }}
 */
export function buildEnvironment(environmentContract) {
  const contract = /** @type {any} */ (environmentContract || {});
  const group = new THREE.Group();
  group.name = "environment";
  const ground = buildGround(contract);
  const grid = buildGrid(contract);
  grid.visible = false;
  group.add(ground, grid);
  group.add(...buildLighting());
  return { group, grid };
}

/**
 * @param {any} contract
 */
function buildGround(contract) {
  const geometry = new THREE.PlaneGeometry(GROUND_SIZE_METRES, GROUND_SIZE_METRES);
  // Plane is XY by default; lay it flat so it becomes the Y=0 ground.
  geometry.rotateX(-Math.PI / 2);
  const material = new THREE.MeshStandardMaterial({
    color: colorFromArray(contract.ground_color || [0.05, 0.06, 0.08]),
    roughness: 0.95,
    metalness: 0.0,
  });
  const ground = new THREE.Mesh(geometry, material);
  ground.receiveShadow = true;
  ground.name = "ground";
  return ground;
}

/**
 * @param {any} contract
 */
function buildGrid(contract) {
  const gridColor = colorFromArray(contract.grid_color || [0.04, 0.62, 0.68]);
  const grid = new THREE.GridHelper(GROUND_SIZE_METRES, GRID_DIVISIONS, gridColor, gridColor);
  const material = /** @type {THREE.Material} */ (grid.material);
  material.transparent = true;
  material.opacity = 0.25;
  grid.name = "debug-grid";
  return grid;
}

/**
 * @returns {THREE.Object3D[]}
 */
function buildLighting() {
  const ambient = new THREE.AmbientLight(0xffffff, AMBIENT_INTENSITY);
  const keyLight = new THREE.DirectionalLight(0xffffff, KEY_LIGHT_INTENSITY);
  keyLight.position.set(6, 12, 8);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  keyLight.shadow.camera.near = 1;
  keyLight.shadow.camera.far = 60;
  keyLight.shadow.camera.left = -20;
  keyLight.shadow.camera.right = 20;
  keyLight.shadow.camera.top = 20;
  keyLight.shadow.camera.bottom = -20;
  keyLight.shadow.bias = -0.0008;
  const fillLight = new THREE.DirectionalLight(0xbcd4ff, FILL_LIGHT_INTENSITY);
  fillLight.position.set(-8, 6, -4);
  return [ambient, keyLight, fillLight];
}
