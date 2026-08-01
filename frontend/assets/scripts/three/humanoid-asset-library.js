// @ts-check

import * as THREE from "../../vendor/three/three.module.js";
import { GLTFLoader } from "./loaders/GLTFLoader.js";
import { clone as cloneSkinnedScene } from "./utils/SkeletonUtils.js";
import { colorFromArray, clamp } from "./scene-space.js";

const HUMAN_MODEL_URLS = [
  "/assets/models/humans/reconstruction-human.glb",
  "/assets/models/humans/reconstruction-human-female.glb",
];
const HUMAN_ANIMATION_URL = "/assets/models/animations/human-base-animations.glb";
const MODEL_HEIGHT_METRES = 1.75;
const MINIMUM_MODEL_HEIGHT = 0.2;
const DEFAULT_STRIDE_METRES = 1.25;
const MOTION_CLIP_NAMES = {
  idle: "Idle_A",
  walk: "Walk",
  brisk_walk: "Jog",
  run: "Sprint",
};
const ROOT_BONE_PREFIX = "root.";
const FOOT_BONE_NAMES = ["foot_l", "foot_r"];

// The dashboard can show more than one saved result. Keep one immutable source
// bundle in memory and clone it per actor; otherwise each result card downloads
// and parses the same 11 MB of geometry and clips again.
let sharedAssetBundlePromise = null;

/**
 * Loads and clones the browser humanoid assets once per page. A clone gets its own
 * skeleton and AnimationMixer, so one track can never deform another track's actor.
 */
export class HumanoidAssetLibrary {
  constructor() {
    this._ready = null;
    this._models = [];
    this._clips = new Map();
  }

  /** @returns {Promise<boolean>} */
  preload() {
    if (!sharedAssetBundlePromise) sharedAssetBundlePromise = loadSharedAssetBundle();
    if (!this._ready) {
      this._ready = sharedAssetBundlePromise.then((bundle) => {
        this._models = bundle.models;
        this._clips = bundle.clips;
        return bundle.ready;
      });
    }
    return this._ready;
  }

  /**
   * @param {any} entity
   * @returns {{ group: THREE.Object3D, mixer: THREE.AnimationMixer, actions: Map<string, THREE.AnimationAction>, height: number, strideLength: number, clipDuration: number }}
   */
  buildActor(entity) {
    if (!this._models.length || !this._clips.size) {
      throw new Error("Humanoid assets are not ready");
    }
    const model = this._models[modelIndex(entity)] || this._models[0];
    const group = cloneSkinnedScene(model.scene);
    group.name = `humanoid-${entity.track_id}`;
    const height = placeOnGround(group, targetHeight(entity));
    styleActor(group, entity.appearance || {});
    const mixer = new THREE.AnimationMixer(group);
    const actions = buildActions(mixer, this._clips);
    const requestedClip = String(entity.motion_profile?.clip || "idle");
    const activeClip = actions.has(requestedClip) ? requestedClip : "idle";
    const action = actions.get(activeClip);
    if (action) action.play();
    const clipDuration = this._clips.get(activeClip)?.duration || 1;
    return {
      group,
      mixer,
      actions,
      height,
      strideLength: Math.max(DEFAULT_STRIDE_METRES, height * 0.72),
      clipDuration,
      footBones: findFootBones(group),
    };
  }
}

/** @returns {Promise<{models: any[], clips: Map<string, THREE.AnimationClip>, ready: boolean}>} */
async function loadSharedAssetBundle() {
  try {
    const loader = new GLTFLoader();
    const load = (url) => new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, reject);
    });
    const [male, female, animationPack] = await Promise.all([
      load(HUMAN_MODEL_URLS[0]),
      load(HUMAN_MODEL_URLS[1]),
      load(HUMAN_ANIMATION_URL),
    ]);
    const clips = new Map();
    for (const clip of animationPack.animations || []) {
      const motionName = Object.entries(MOTION_CLIP_NAMES)
        .find(([, clipName]) => clipName === clip.name)?.[0];
      if (motionName) clips.set(motionName, withoutRootTracks(clip));
    }
    const models = [male, female];
    const ready = models.every(hasSkinnedMesh) && clips.has("idle") && clips.has("walk");
    return { models: ready ? models : [], clips: ready ? clips : new Map(), ready };
  } catch (error) {
    console.warn("Three.js humanoid assets could not be loaded; using the safe fallback actor.", error);
    return { models: [], clips: new Map(), ready: false };
  }
}

/** @param {THREE.AnimationMixer} mixer @param {Map<string, THREE.AnimationClip>} clips */
function buildActions(mixer, clips) {
  const actions = new Map();
  clips.forEach((clip, name) => actions.set(name, mixer.clipAction(clip)));
  return actions;
}

/** @param {THREE.AnimationClip} clip @returns {THREE.AnimationClip} */
function withoutRootTracks(clip) {
  const copy = clip.clone();
  copy.tracks = copy.tracks.filter((track) => {
    const name = String(track.name);
    // The evidence path owns world translation and heading. Keeping the mocap
    // root transform would apply a second displacement/turn and cause popping.
    return !name.startsWith(ROOT_BONE_PREFIX);
  });
  return copy;
}

/** @param {any} asset @returns {boolean} */
function hasSkinnedMesh(asset) {
  let found = false;
  asset.scene?.traverse((object) => { if (object.isSkinnedMesh) found = true; });
  return found;
}

/** @param {THREE.Object3D} group @returns {THREE.Bone[]} */
function findFootBones(group) {
  const bones = [];
  group.traverse((object) => {
    if (object.isBone && FOOT_BONE_NAMES.includes(object.name)) bones.push(object);
  });
  return bones;
}

/** @param {any} entity @returns {number} */
function modelIndex(entity) {
  const seed = Number(entity.appearance_seed) || 0;
  return Math.abs(Math.trunc(seed)) % HUMAN_MODEL_URLS.length;
}

/** @param {any} entity @returns {number} */
function targetHeight(entity) {
  const measured = Number(entity.proxy?.dimensions_metres?.[2]);
  return measured > MINIMUM_MODEL_HEIGHT ? measured : MODEL_HEIGHT_METRES;
}

/** @param {THREE.Object3D} group @param {number} height @returns {number} */
function placeOnGround(group, height) {
  const bounds = new THREE.Box3().setFromObject(group);
  const sourceHeight = Math.max(MINIMUM_MODEL_HEIGHT, bounds.max.y - bounds.min.y);
  const scale = height / sourceHeight;
  group.scale.setScalar(scale);
  group.position.y -= bounds.min.y * scale;
  return height;
}

/** @param {THREE.Object3D} group @param {any} appearance */
function styleActor(group, appearance) {
  const upper = colorFromArray(appearance.upper_color || [0.10, 0.38, 0.50]);
  const lower = colorFromArray(appearance.lower_color || [0.12, 0.16, 0.22]);
  group.traverse((object) => {
    const mesh = /** @type {THREE.Mesh} */ (object);
    if (!mesh.isMesh || !mesh.material) return;
    const hasMaterialArray = Array.isArray(mesh.material);
    const materials = hasMaterialArray ? mesh.material : [mesh.material];
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const styledMaterials = materials.map((material) => {
      const materialName = String(material.name || "").toLowerCase();
      const lowerBody = /joint|lower|pant|shoe|foot/.test(materialName);
      const styled = new THREE.MeshStandardMaterial({
        // The source asset uses a palette atlas whose alpha channel is not reliable
        // across browser image decoders. A clean lit material is more readable and
        // deterministic than a textured mesh that can disappear on one GPU/browser.
        color: lowerBody ? lower : upper,
        map: null,
        roughness: 0.86,
        metalness: 0.0,
        side: THREE.DoubleSide,
      });
      return styled;
    });
    mesh.material = hasMaterialArray ? styledMaterials : styledMaterials[0];
  });
}
