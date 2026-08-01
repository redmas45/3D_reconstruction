// @ts-check
// The reconstruction viewer: owns the WebGL renderer, camera, and the currently-selected
// gap's scene, and exposes the control surface the dashboard drives (play/pause/restart,
// gap selection, uncertainty/debug toggles, capture). Only the selected gap is built, and
// its geometry is disposed before the next is built, so switching gaps neither leaks GPU
// memory nor rebuilds gaps the viewer is not showing.

import * as THREE from "../../vendor/three/three.module.js";
import { buildSceneCamera, applyCameraContract } from "./scene-camera.js";
import { buildEnvironment } from "./scene-environment.js";
import { buildActor } from "./actor-factory.js";
import { HumanoidAssetLibrary } from "./humanoid-asset-library.js";
import { buildVehicle } from "./vehicle-factory.js";
import {
  buildPathCurve, pathPose, animateWalk, updateHumanoidMotion, animateWheels, gaitPhase,
} from "./motion-system.js";
import {
  applyConfidenceOpacity, buildUncertaintyCorridor, buildGroundingDisc,
} from "./confidence-visuals.js";
import { buildBackplateTexture } from "./backplate.js";
import { disposeObject, clearChildren } from "./resource-disposal.js";
import { webglIsAvailable, createWebglFallbackNotice } from "./webgl-fallback.js";
import { CaptureController, downloadCapture } from "./capture-controller.js";

const MAXIMUM_PIXEL_RATIO = 2;
const INTEGRATED_GPU_PIXEL_RATIO = 1.4;
const LOOP_SECONDS_FLOOR = 3.5;
const WALK_MOVING_SPEED = 0.2;
const MINIMUM_ANCHOR_SCALE = 0.55;
const MAXIMUM_ANCHOR_SCALE = 1.8;

export class ReconstructionView {
  /**
   * @param {HTMLElement} container
   * @param {{ overlay?: boolean, showUncertainty?: boolean }} [options]
   */
  constructor(container, options = {}) {
    this._container = container;
    this._overlayMode = Boolean(options.overlay);
    this._showUncertainty = options.showUncertainty ?? !this._overlayMode;
    this._manifest = null;
    this._gapIndex = 0;
    this._playing = false;
    this._debug = false;
    this._animated = [];
    this._clock = new THREE.Clock();
    this._elapsed = 0;
    this._frameHandle = 0;
    this._capture = null;
    this._humanoidAssets = new HumanoidAssetLibrary();
    this._assetReady = Promise.resolve(false);
    this._sceneBuildToken = 0;
    this._available = webglIsAvailable();
    if (this._available) this._initRenderer();
  }

  /** @returns {boolean} */
  get isAvailable() {
    return this._available;
  }

  _initRenderer() {
    this._renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: this._overlayMode,
      powerPreference: "high-performance",
    });
    if (this._overlayMode) this._renderer.setClearColor(0x000000, 0);
    this._renderer.outputColorSpace = THREE.SRGBColorSpace;
    this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this._renderer.toneMappingExposure = 1.05;
    this._renderer.shadowMap.enabled = true;
    this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this._renderer.setPixelRatio(this._targetPixelRatio());
    this._canvas = this._renderer.domElement;
    this._canvas.className = "three-canvas";
    this._container.appendChild(this._canvas);
    this._scene = new THREE.Scene();
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(this._container);
  }

  /**
   * Cap the device pixel ratio; integrated GPUs get a lower cap to protect frame rate.
   * @returns {number}
   */
  _targetPixelRatio() {
    const cap = this._isIntegratedGpu() ? INTEGRATED_GPU_PIXEL_RATIO : MAXIMUM_PIXEL_RATIO;
    return Math.min(window.devicePixelRatio || 1, cap);
  }

  _isIntegratedGpu() {
    try {
      const gl = this._renderer.getContext();
      const info = gl.getExtension("WEBGL_debug_renderer_info");
      const name = info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : "";
      return /intel|software|swiftshader|llvmpipe|apple gpu/i.test(name);
    } catch (error) {
      return false;
    }
  }

  /**
   * Load a validated manifest and build the first gap.
   * @param {object} manifest
   */
  load(manifest) {
    this._manifest = /** @type {any} */ (manifest);
    if (!this._available) {
      this._container.appendChild(createWebglFallbackNotice());
      return;
    }
    const source = this._manifest.source || {};
    this._sourceAspect = (Number(source.width) || 16) / (Number(source.height) || 9);
    this._camera = buildSceneCamera(this._manifest.camera_default, this._sourceAspect);
    this._assetReady = this._humanoidAssets.preload();
    void this.selectGap(0);
    this._resize();
    if (!this._overlayMode) this.play();
  }

  /**
   * Build a single gap's scene. Any previously-built gap is disposed first.
   * @param {number} index
   */
  async selectGap(index) {
    if (!this._available || !this._manifest) return;
    const gaps = this._manifest.gaps || [];
    if (index < 0 || index >= gaps.length) return;
    this._gapIndex = index;
    const buildToken = ++this._sceneBuildToken;
    const assetsAvailable = await this._assetReady;
    if (buildToken !== this._sceneBuildToken || !this._manifest) return;
    this._teardownGap();
    const gap = gaps[index];
    this._scene.background = this._overlayMode
      ? null
      : buildBackplateTexture(gap.environment);
    const environment = buildEnvironment(gap.environment);
    this._environment = environment.group;
    this._grid = environment.grid;
    this._grid.visible = this._debug;
    if (this._overlayMode) this._hideOverlayGround();
    this._scene.add(this._environment);
    applyCameraContract(this._camera, gap.camera, this._sourceAspect);
    this._buildEntities(gap, assetsAvailable);
    this._elapsed = 0;
    this._loopSeconds = Math.max(LOOP_SECONDS_FLOOR, Number(gap.duration_seconds) || LOOP_SECONDS_FLOOR);
    this._renderOnce();
  }

  /**
   * @param {any} gap
   */
  _buildEntities(gap, assetsAvailable) {
    this._animated = [];
    (gap.entities || []).forEach((entity) => {
      const animatedEntity = this._buildEntity(entity, assetsAvailable);
      if (animatedEntity) this._scene.add(animatedEntity.root);
    });
  }

  /**
   * @param {any} entity
   * @returns {{ root: THREE.Group } | null}
   */
  _buildEntity(entity, assetsAvailable) {
    const path = buildPathCurve(entity);
    const isVehicle = entity.category === "vehicle";
    const model = isVehicle
      ? buildVehicle(entity)
      : buildHumanoidOrFallback(this._humanoidAssets, entity, assetsAvailable);
    const actorScale = applyVisualAnchorScale(
      model,
      entity,
      pathPose(path.curve, 0).position,
      this._camera,
      Number(this._manifest.source?.height) || 0,
    );
    if (Number.isFinite(actorScale) && Number(model.strideLength) > 0) {
      model.strideLength *= actorScale;
    }
    const confidence = Number(entity.confidence) || 0;
    applyConfidenceOpacity(model.group, confidence, entity.visual_fidelity_tier);

    const root = new THREE.Group();
    root.name = `entity-${entity.track_id}`;
    root.add(model.group);
    if (this._showUncertainty) {
      const corridor = buildUncertaintyCorridor(
        path.curve, entity.uncertainty?.position_radius_metres, confidence,
      );
      if (corridor) root.add(corridor);
    }
    if (!this._overlayMode) root.add(buildGroundingDisc(0.4, confidence));

    const animated = {
      root,
      entity,
      isVehicle,
      curve: path.curve,
      length: path.length,
      joints: isVehicle ? null : /** @type {any} */ (model).joints,
      wheels: isVehicle ? /** @type {any} */ (model).wheels : null,
      wheelRadius: isVehicle ? /** @type {any} */ (model).wheelRadius : 0,
      strideLength: isVehicle ? 1 : /** @type {any} */ (model).strideLength,
      phaseOffset: entity.motion_profile?.phase_offset || 0,
      cadence: entity.motion_profile?.cadence_scale || 1,
      speed: entity.motion_profile?.speed_meters_per_second || 0,
      mixer: isVehicle ? null : /** @type {any} */ (model).mixer || null,
      clipDuration: isVehicle ? 0 : /** @type {any} */ (model).clipDuration || 0,
      footBones: isVehicle ? null : /** @type {any} */ (model).footBones || [],
      model: model.group,
      previousDistance: 0,
    };
    this._animated.push(animated);
    // Seat the entity at the start of its path immediately.
    this._updateEntity(animated, 0);
    return animated;
  }

  /**
   * @param {any} animated @param {number} t
   */
  _updateEntity(animated, t) {
    const pose = pathPose(animated.curve, t);
    animated.root.position.copy(pose.position);
    // The model is the first child; keep the corridor/disc unrotated by rotating the model.
    animated.root.position.y = 0;
    const model = animated.root.children[0];
    model.position.copy(pose.position).setY(0).sub(animated.root.position);
    model.rotation.y = pose.yaw;

    const distance = animated.length * t;
    let delta = distance - animated.previousDistance;
    if (delta < 0) delta = 0; // loop wrapped
    animated.previousDistance = distance;

    if (animated.isVehicle) {
      animateWheels(animated.wheels, delta, animated.wheelRadius);
      return;
    }
    if (animated.mixer) {
      updateHumanoidMotion(animated, distance, this._elapsed);
      return;
    }
    const phase = gaitPhase(distance, animated.strideLength, animated.phaseOffset, animated.cadence);
    animateWalk(animated.joints, {
      phase,
      moving: animated.speed >= WALK_MOVING_SPEED,
      intensity: 1,
    });
  }

  play() {
    if (!this._available || this._playing) return;
    this._playing = true;
    this._clock.start();
    this._tick();
  }

  pause() {
    this._playing = false;
    if (this._frameHandle) cancelAnimationFrame(this._frameHandle);
    this._frameHandle = 0;
  }

  restart() {
    this._elapsed = 0;
    this._animated.forEach((animated) => { animated.previousDistance = 0; });
    if (!this._playing) this._renderOnce();
  }

  _tick() {
    if (!this._playing) return;
    this._frameHandle = requestAnimationFrame(() => this._tick());
    this._elapsed += this._clock.getDelta();
    const t = (this._elapsed % this._loopSeconds) / this._loopSeconds;
    if (t < 0.0005) this._animated.forEach((animated) => { animated.previousDistance = 0; });
    this._animated.forEach((animated) => this._updateEntity(animated, t));
    this._renderer.render(this._scene, this._camera);
  }

  _renderOnce() {
    if (this._available && this._camera) this._renderer.render(this._scene, this._camera);
  }

  /**
   * Render the selected gap at the same clock position as the source video.
   * @param {number} secondsFromGapStart
   */
  setTimelineTime(secondsFromGapStart) {
    if (!this._available || !this._manifest) return;
    const loopSeconds = Math.max(LOOP_SECONDS_FLOOR, this._loopSeconds || LOOP_SECONDS_FLOOR);
    const boundedSeconds = Math.max(0, Math.min(Number(secondsFromGapStart) || 0, loopSeconds));
    this._elapsed = boundedSeconds;
    const t = boundedSeconds / loopSeconds;
    this._animated.forEach((animated) => this._updateEntity(animated, t));
    this._renderOnce();
  }

  /** @param {boolean} enabled */
  setDebug(enabled) {
    this._debug = enabled;
    if (this._grid) this._grid.visible = enabled;
    this._renderOnce();
  }

  /** @param {boolean} enabled */
  setUncertainty(enabled) {
    this._showUncertainty = enabled;
    if (this._manifest) this.selectGap(this._gapIndex);
  }

  /** @returns {boolean} */
  captureSupported() {
    return CaptureController.isSupported();
  }

  startCapture() {
    if (!this._available) return;
    this._capture = new CaptureController(this._canvas, 30);
    this._capture.start();
  }

  /**
   * @param {string} filename
   * @returns {Promise<void>}
   */
  async stopCapture(filename) {
    if (!this._capture) return;
    const blob = await this._capture.stop();
    this._capture = null;
    downloadCapture(blob, filename);
  }

  _resize() {
    if (!this._available || !this._camera) return;
    const width = this._container.clientWidth || 640;
    const height = this._container.clientHeight || 360;
    const fitted = fitToAspect(width, height, this._sourceAspect || width / height);
    this._renderer.setSize(fitted.width, fitted.height, true);
    this._renderOnce();
  }

  _teardownGap() {
    this._animated.forEach((animated) => disposeObject(animated.root));
    this._animated = [];
    if (this._environment) disposeObject(this._environment);
    clearChildren(this._scene);
    if (this._scene.background && this._scene.background.isTexture) {
      this._scene.background.dispose();
      this._scene.background = null;
    }
  }

  _hideOverlayGround() {
    if (!this._environment) return;
    this._environment.traverse((object) => {
      if (object.name === "ground") object.visible = false;
    });
  }

  dispose() {
    this._sceneBuildToken += 1;
    this.pause();
    this._teardownGap();
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this._renderer) {
      this._renderer.dispose();
      if (this._canvas && this._canvas.parentNode) this._canvas.parentNode.removeChild(this._canvas);
    }
  }
}

/**
 * Assets are intentionally optional at runtime: a failed GLB load should produce a
 * readable procedural fallback rather than prevent the source video from playing.
 * @param {HumanoidAssetLibrary} library
 * @param {any} entity
 * @param {boolean} assetsAvailable
 */
function buildHumanoidOrFallback(library, entity, assetsAvailable) {
  if (entity.visual_fidelity_tier === "weak") return buildActor(entity);
  if (assetsAvailable) {
    try {
      return library.buildActor(entity);
    } catch (error) {
      console.warn("Humanoid clone failed; using the fallback actor for this track.", error);
    }
  }
  return buildActor(entity);
}

/**
 * Match a rigged actor's projected height to the last visible boundary bbox. This
 * corrects the common monocular-camera failure where a catalog-height person is
 * physically plausible but visibly too large or too small in the plate.
 * @param {{ group: THREE.Object3D, height?: number }} model
 * @param {any} entity
 * @param {THREE.Vector3} basePosition
 * @param {THREE.Camera} camera
 * @param {number} sourceHeight
 * @returns {number}
 */
function applyVisualAnchorScale(model, entity, basePosition, camera, sourceHeight) {
  const anchor = entity.visual_anchor;
  const targetFraction = Number(anchor?.height_fraction);
  const baseHeight = Number(model.height);
  if (!anchor || !Number.isFinite(targetFraction) || targetFraction <= 0
    || !Number.isFinite(baseHeight) || baseHeight <= 0 || sourceHeight <= 0) return 1;
  const base = basePosition.clone();
  const top = base.clone().setY(base.y + baseHeight);
  const projectedBase = base.project(camera);
  const projectedTop = top.project(camera);
  const projectedFraction = Math.abs(projectedTop.y - projectedBase.y) / 2;
  if (!Number.isFinite(projectedFraction) || projectedFraction <= 0) return 1;
  const desiredFraction = Math.min(1, targetFraction);
  const scale = THREE.MathUtils.clamp(
    desiredFraction / projectedFraction,
    MINIMUM_ANCHOR_SCALE,
    MAXIMUM_ANCHOR_SCALE,
  );
  model.group.scale.multiplyScalar(scale);
  return scale;
}

/**
 * Fit a source aspect ratio inside a container, letterboxing so the calibrated projection
 * is preserved rather than stretched.
 * @param {number} containerWidth @param {number} containerHeight @param {number} aspect
 * @returns {{ width: number, height: number }}
 */
function fitToAspect(containerWidth, containerHeight, aspect) {
  let width = containerWidth;
  let height = Math.round(width / aspect);
  if (height > containerHeight) {
    height = containerHeight;
    width = Math.round(height * aspect);
  }
  return { width: Math.max(1, width), height: Math.max(1, height) };
}
