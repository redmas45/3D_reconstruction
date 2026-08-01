// @ts-check
// Loads and defensively validates the scene manifest before the renderer trusts it.
//
// The backend already validates the manifest at build time; this is zero-trust ingestion
// at the browser boundary (a malformed or truncated payload must fail loudly, not draw
// broken geometry). It never invents data: missing optional fields get conservative
// defaults, but a structurally invalid manifest is rejected.

const SCENE_MANIFEST_SCHEMA_VERSION = 1;
const RENDERER_NAME = "threejs";

export class SceneManifestError extends Error {}

/**
 * Fetch the manifest for a completed job from the local backend.
 * @param {string} url
 * @returns {Promise<object>}
 */
export async function fetchSceneManifest(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new SceneManifestError("The scene manifest could not be loaded");
  const payload = await response.json();
  return parseSceneManifest(payload);
}

/**
 * Validate an already-parsed manifest object.
 * @param {unknown} payload
 * @returns {object}
 */
export function parseSceneManifest(payload) {
  if (!payload || typeof payload !== "object") {
    throw new SceneManifestError("Scene manifest is not an object");
  }
  const manifest = /** @type {any} */ (payload);
  if (manifest.schema_version !== SCENE_MANIFEST_SCHEMA_VERSION) {
    throw new SceneManifestError("Unsupported scene manifest schema version");
  }
  if (manifest.renderer !== RENDERER_NAME) {
    throw new SceneManifestError("Scene manifest is not addressed to the Three.js renderer");
  }
  if (!manifest.source || typeof manifest.source !== "object") {
    throw new SceneManifestError("Scene manifest is missing its source block");
  }
  if (!Array.isArray(manifest.gaps)) {
    throw new SceneManifestError("Scene manifest is missing its gaps");
  }
  manifest.gaps.forEach(validateGap);
  return manifest;
}

/**
 * @param {any} gap @param {number} index
 */
function validateGap(gap, index) {
  if (!gap || typeof gap !== "object") {
    throw new SceneManifestError(`Gap ${index} is not an object`);
  }
  if (!gap.camera || typeof gap.camera !== "object") {
    throw new SceneManifestError(`Gap ${index} is missing a camera`);
  }
  if (!Array.isArray(gap.entities)) {
    throw new SceneManifestError(`Gap ${index} is missing its entities`);
  }
  gap.entities.forEach((entity, entityIndex) => validateEntity(entity, index, entityIndex));
}

/**
 * @param {any} entity @param {number} gapIndex @param {number} entityIndex
 */
function validateEntity(entity, gapIndex, entityIndex) {
  if (!entity || typeof entity !== "object") {
    throw new SceneManifestError(`Gap ${gapIndex} entity ${entityIndex} is not an object`);
  }
  if (!Array.isArray(entity.waypoints) || entity.waypoints.length < 3) {
    throw new SceneManifestError(
      `Gap ${gapIndex} entity ${entityIndex} must carry at least three waypoints`,
    );
  }
  const everyWaypointHasWorld = entity.waypoints.every(
    (waypoint) => waypoint && Array.isArray(waypoint.world) && waypoint.world.length >= 3,
  );
  if (!everyWaypointHasWorld) {
    throw new SceneManifestError(
      `Gap ${gapIndex} entity ${entityIndex} has a waypoint without world coordinates`,
    );
  }
}
