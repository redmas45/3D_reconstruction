// @ts-check
// Frees GPU resources when a gap is torn down. Three.js does not garbage-collect
// geometries, materials, or textures; leaving them allocated leaks memory every time
// the viewer switches gaps. disposeObject() walks a subtree and releases everything.

import * as THREE from "../../vendor/three/three.module.js";

/**
 * Dispose every geometry, material, and texture beneath (and including) an object, then
 * detach it from its parent. Safe to call on a group or a single mesh.
 * @param {THREE.Object3D | null | undefined} root
 */
export function disposeObject(root) {
  if (!root) return;
  root.traverse((node) => {
    const mesh = /** @type {THREE.Mesh} */ (node);
    if (mesh.geometry && typeof mesh.geometry.dispose === "function") {
      mesh.geometry.dispose();
    }
    disposeMaterial(mesh.material);
  });
  if (root.parent) root.parent.remove(root);
}

/**
 * @param {THREE.Material | THREE.Material[] | null | undefined} material
 */
export function disposeMaterial(material) {
  if (!material) return;
  const materials = Array.isArray(material) ? material : [material];
  materials.forEach((entry) => {
    if (!entry) return;
    Object.values(entry).forEach((value) => {
      if (value && value.isTexture && typeof value.dispose === "function") {
        value.dispose();
      }
    });
    if (typeof entry.dispose === "function") entry.dispose();
  });
}

/**
 * Remove and dispose every child of a container without disposing the container itself.
 * @param {THREE.Object3D} container
 */
export function clearChildren(container) {
  for (let index = container.children.length - 1; index >= 0; index -= 1) {
    disposeObject(container.children[index]);
  }
}
