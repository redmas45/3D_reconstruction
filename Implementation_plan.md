# Blender Forensic Reconstruction — Implementation Plan

## 1. Document Purpose

This is the approved, living implementation plan for the AI evidence-gap reconstruction project. It records the target architecture, acceptance gates, completed work, measured results, and remaining validation work.

### Implementation status — 21 July 2026

- Phase 0 is implemented: the OpenCV 2.5D renderer remains an explicit fallback, Blender 4.5 LTS runs headlessly through a JSON-only boundary, and real preview/production timings are recorded.
- Phase 1 is implemented and unit tested: plan v2, visible-only evidence validation, delayed hidden-truth materialization, global identity registry, static/dynamic camera measurement, robust height priors, forward-predicted three-point paths, soft post-gap residuals, crossing-appearance rejection, heading-conflict downgrades, and formal presentation filtering.
- Phase 2 is implemented and visually reviewed: calibrated forensic camera, ground plane, city-street proxies, evidence inset, procedural people/vehicles, confidence HUD, and midpoint rendering.
- Phase 3 is implemented and technically verified on gap 0 and a short end-to-end fixture: articulated motion, lifecycle fades, confidence-to-fidelity tiers, forensic entry/exit shutter, exact frame count, exact fractional frame rate, and full-resolution Blender encoding.
- UI/pipeline integration is implemented: Blender is the default selectable renderer, 2.5D is an explicit fallback, errors do not silently switch renderers, progress is surfaced through the existing queue, FFmpeg preserves source audio, and final media contracts are validated.
- Arbitrary-video hardening now covers 59.94 fps sources without Blender's integer FPS clamp, uses three configurable parallel Blender gap workers, and avoids writing unused raw visible-segment copies. Full videos remain streamed so the 32 GB workstation retains memory headroom for YOLO, Blender, and the operating system.
- Windows job metadata persistence uses unique temporary files and bounded retry/backoff so transient antivirus or indexing locks cannot fail a reconstruction; instant preparation stages emit one persisted update instead of a rapid write burst.
- Active and queued jobs can be cancelled from the UI. Cancellation is propagated through Python checkpoints and terminates active Blender and FFmpeg subprocesses; cancelled jobs remain removable but are never mislabeled as failures.
- Phase 4 (three representative gaps), Phase 5 (full `input_vid3.mp4`), Phase 6 final judge polish, and Phase 7 second-video validation remain approval gates. They are not represented as complete.

Measured verification artifacts:

- `outputs/blender_preview_input_vid3/gap_00/`: original-video gap-0 midpoint, animation, plan, `.blend`, report, log, and contact sheet.
- `outputs/e2e_blender_smoke/`: 150-frame UI-pipeline smoke result with one 38-frame Blender gap, H.264 video, AAC audio, and exact 1280×720 / 29.970029 fps contract.
- Eevee production timing on this CPU: final 57-frame gap with transitions in 560.489 seconds; 38-frame smoke gap in 261.256 seconds.
- Automated suite: 35 tests passing after the Blender/UI/audio integration and residual-confidence correction.

## 2. Current State

The project now:

- reads videos from `data/input`
- distributes approximately 25% of the video across random 1–3 second gaps
- keeps the remaining 75% as YOLO-annotated evidence
- tracks visible people, vehicles, and carried objects
- writes strict plan-v2, identity-registry, camera-motion, selection, and render-report contracts
- renders inferred gaps with headless Blender forensic 3D by default
- retains OpenCV 2.5D only as an operator-selected fallback
- prevents hidden-frame detections and does not materialize hidden truth until every inferred gap is rendered
- evaluates completed gaps only after rendering
- stitches the exact original frame count and uses FFmpeg to preserve source audio

The earlier OpenCV output had background dissolves, duplicated pedestrians, sliding cutouts, weak masks, and no scene geometry. That output remains available only as a compatibility fallback. The current Blender direction uses deliberate stylized forensic 3D and exposes uncertainty rather than attempting an untrustworthy photoreal recovery.

Blender 4.5.10 LTS and FFmpeg 8.1.2 are installed and integrated through checked-in scripts. Blender reads validated JSON contracts and does not import the system-Python business modules.

## 3. Product Goal

The target output is a professional forensic visualization:

1. Visible footage remains original evidence with YOLO classifications.
2. Approximately 25% of the video remains hidden across short 1–3 second gaps.
3. Every hidden gap switches to a clearly identified 3D inference scene.
4. The 3D scene uses evidence-derived people, vehicles, paths, directions, colors, and confidence.
5. People have real walking or stationary animation instead of sliding cutouts.
6. Vehicles move and orient naturally along the inferred path.
7. Camera perspective, ground contact, scale, lighting, shadows, and occlusion are coherent.
8. Uncertainty is visible and the output never claims to be recovered ground truth.
9. The output retains the source duration, frame rate, resolution, and audio when available.
10. The entire workflow remains reproducible from the local UI started with `python app.py`.

The intended label is:

```text
AI-INFERRED FORENSIC RECONSTRUCTION — NOT GROUND TRUTH
```

## 4. Recommended Visual Direction

### Default: deliberate forensic 3D

The default renderer should use a clean, intentional 3D visual language:

- neutral dark environment
- perspective-matched ground plane
- simplified building and street proxy geometry
- rigged human figures
- simplified but recognizable vehicle models
- evidence-derived clothing and object colors
- soft shadows and controlled lighting
- confidence paths, labels, and timecode
- original evidence frame used as a contextual backplate or optional side panel

This is recommended over attempting photorealism. A coherent stylized reconstruction will look professional and honest. A failed photorealistic reconstruction looks like a visual defect or hallucination.

### Optional mode: evidence-camera overlay

An optional renderer may place 3D figures over a stabilized evidence backplate. This mode should not be the default until foreground removal and scene depth are reliable.

### Explicit non-goal

The system will not claim that the inferred scene is historically exact. A single camera cannot reveal entities or actions that appear only inside the hidden interval.

## 5. Why Blender

Blender provides the parts missing from the OpenCV renderer:

- real 3D coordinates and cameras
- armatures and skeletal animation
- reusable procedural meshes
- walk cycles and keyframe interpolation
- materials and evidence-derived colors
- lights and physically coherent shadows
- depth ordering and occlusion
- vehicle orientation and wheel animation
- deterministic Python automation through `bpy`
- headless rendering for repeatable command-line runs

Blender does not improve inference by itself. It renders the reconstruction plan. Tracking, identity continuity, path inference, and uncertainty must remain separate and testable.

Blender MCP is not required for the production pipeline. Checked-in Python scripts executed through Blender background mode will be the authoritative implementation. MCP could be used later for optional interactive scene development.

## 6. Evidence Contract

The hidden ground-truth frames must remain unavailable to every reconstruction stage.

Allowed inputs for a hidden gap:

- detections from visible ranges
- tracks from visible ranges
- visible frames before the gap
- visible frames after the gap
- appearance samples from visible frames
- inferred camera and ground-plane parameters
- optional witness statement stored separately from video evidence

Forbidden inputs until evaluation:

- any hidden frame
- detections generated from hidden ground truth
- optical flow calculated through hidden frames
- appearance crops taken from hidden frames
- metrics from hidden truth used to change the current reconstruction
- the post-gap position used as a hard target to back-solve speed, force arrival, or reshape the predicted path

The post-gap observation is allowed only as a soft consistency check. The primary path must be predicted from pre-gap velocity, heading, acceleration limits, and scene constraints. Evaluation records how far that prediction is from the post-gap observation. The planner must not silently bend the path until that residual becomes zero.

Hidden truth may be read only after every gap has been rendered.

## 7. Target Architecture

```text
Input video
    |
    +-- distributed short-gap selector
    |
    +-- visible-frame detection and tracking
    |
    +-- cross-gap identity matching
    |
    +-- scene intelligence report
    |
    +-- per-gap reconstruction plan v2
            |
            +-- camera and ground-plane calibration
            +-- entity lifecycle and path inference
            +-- appearance/color extraction
            +-- uncertainty model
            |
            +-- Blender scene exporter
                    |
                    +-- scene geometry
                    +-- rigged humans
                    +-- vehicles
                    +-- animation
                    +-- lighting and HUD
                    |
                    +-- Blender background render
                            |
                            +-- gap PNG sequence
                            +-- encoded gap video
                            +-- final stitch and audio mux
                            +-- hidden-truth evaluation
```

Business logic remains in normal Python modules. Blender scripts should build and render scenes, not decide who exists or what happened.

Blender's bundled Python must not import the system `src/` business-logic modules or depend on system site packages. System Python validates evidence and writes plain JSON contracts for plan v2, the global identity registry, calibration, and proxy geometry. Blender reads those contracts with its standard library and owns only `bpy` scene construction and rendering. A smoke test must verify this process and import boundary.

## 8. Reconstruction Plan Version 2

Each gap plan should become a strict, validated renderer contract.

Proposed structure:

```json
{
  "schema_version": 2,
  "gap_index": 0,
  "hidden_range": {"start": 146, "end": 202},
  "fps": 29.97,
  "duration_seconds": 1.90,
  "overall_confidence": 0.71,
  "camera": {
    "mode": "evidence_matched",
    "motion_model": "stabilized_dynamic",
    "calibration_confidence": 0.83,
    "canonical_frame": 145,
    "focal_length_mm": 35.0,
    "position": [0.0, -12.0, 4.5],
    "rotation_degrees": [68.0, 0.0, 0.0],
    "horizon_normalized_y": 0.39,
    "height_prior": {
      "track_count": 6,
      "median_height_pixels": 214.0,
      "median_absolute_deviation": 11.5,
      "stable": true
    }
  },
  "environment": {
    "style": "forensic_3d",
    "ground_color": [0.10, 0.12, 0.14],
    "backplate_frame": 145
  },
  "entities": [
    {
      "id": "person_12",
      "identity_registry_id": "person_12",
      "kind": "person",
      "confidence": 0.78,
      "lifecycle": "continuous",
      "appearance": {
        "upper_color": [0.08, 0.12, 0.18],
        "lower_color": [0.25, 0.22, 0.18],
        "carried_objects": ["bag_3"]
      },
      "animation": {
        "state": "walk",
        "speed_meters_per_second": 1.2,
        "heading_degrees": 86.0,
        "phase_offset": 0.37
      },
      "boundary_evidence": {
        "pre_gap_heading_degrees": 86.0,
        "post_gap_heading_degrees": 91.0,
        "heading_disagreement_degrees": 5.0,
        "post_gap_position_residual_meters": 0.42
      },
      "path_prediction": {
        "method": "catmull_rom",
        "constraint_mode": "forward_prediction",
        "post_gap_observation_role": "soft_consistency_check",
        "waypoints": [
          {"role": "start", "frame": 146, "world": [-2.1, 3.8, 0.0]},
          {"role": "inferred_midpoint", "frame": 174, "world": [-0.9, 3.9, 0.0]},
          {"role": "predicted_end", "frame": 202, "world": [0.2, 3.8, 0.0]}
        ]
      },
      "uncertainty": {
        "position_radius_meters": 0.65,
        "alternative_paths": 2
      }
    }
  ]
}
```

The exporter must reject malformed plans immediately. Blender must never receive unchecked data.

### Path prediction rules

Every rendered path requires at least three ordered waypoints: start, inferred midpoint, and predicted end. The default interpolation is a centripetal Catmull–Rom curve with duplicated endpoints when only three observations are available. The curve implementation and parameterization must be recorded in the plan so Blender and normal Python tests evaluate the same trajectory.

The path is predicted forward from pre-gap evidence. The inferred midpoint blends pre-gap velocity with permitted acceleration and scene-direction constraints. The predicted end is not replaced by the observed post-gap position.

Post-gap evidence produces position, heading, scale, and appearance residuals plus an updated continuity confidence.

Heading disagreement rules:

- `0–30°`: consistent; no continuity penalty
- `31–60°`: plausible turn; reduce continuity confidence and widen the uncertainty corridor
- `61–100°`: ambiguous identity or unobserved turn; require strong appearance evidence and render at reduced fidelity
- greater than `100°`: reject continuity unless an explicit, evidence-supported turn region exists

A post-gap residual above the configured world-distance threshold must remain visible in the report. The planner must not hide it by forcing the curve through the observation.

## 9. Camera and Ground-Plane Calibration

### Goal

Convert image-space detections into stable world-space motion while accounting for camera motion and scale uncertainty.

### Camera model selection

The pipeline must classify each video as either `static_camera` or `stabilized_dynamic_camera` before applying a ground-plane transform.

- `static_camera`: one reviewed calibration may be reused across the video.
- `stabilized_dynamic_camera`: estimate visible-frame camera motion relative to a canonical evidence frame, then maintain a camera-pose track.

A single global homography must never be assumed when measured camera motion exceeds the configured translation, rotation, or scale threshold. Camera motion can be estimated from static background features using robust feature matching or ECC. Dynamic foreground boxes must be excluded from camera-motion estimation.

Inside a hidden gap, camera pose is predicted from pre-gap camera velocity. The post-gap pose is a consistency observation and contributes to calibration confidence; it is not used to conceal an implausible prediction.

### Ground and scale method

1. Estimate the horizon or vanishing region from visible street lines when reliable.
2. Use the bottom-center of each bounding box as the approximate ground-contact point.
3. Map ground-contact pixels into normalized ground coordinates with the calibration valid for that frame or canonical camera pose.
4. Build height priors from multiple stable observations of the same track.
5. Reject truncated, heavily occluded, low-confidence, and statistically inconsistent bounding boxes.
6. Smooth camera parameters only across evidence frames that pass calibration checks.
7. Store calibration inputs, residuals, and confidence in the reconstruction plan.

### Robust height prior

For each eligible person track:

1. Collect bounding-box heights from visible evidence.
2. Reject boxes touching the image boundary or overlapping another person above the occlusion threshold.
3. Compute the median height and median absolute deviation (MAD).
4. Reject observations farther than `3 × MAD` from the track median.
5. Require at least five accepted observations from the same track by default.
6. Compute robust relative spread as `MAD / median_height` and flag the track as scale-unstable when it exceeds the default `0.15` threshold.

Children, seated people, and unusual poses must not be mixed into a generic adult-height prior without an explicit classification. A scale-unstable track may still be rendered, but it receives a wider size-uncertainty range.

### Calibration confidence

Calibration confidence is separate from entity confidence. It combines:

- static-feature inlier ratio
- camera-motion model residual
- horizon/vanishing-point stability
- number of stable height-prior tracks
- robust height variance
- ground-contact reprojection error

Every component is normalized to `[0, 1]`, where `1` means reliable. The default score is:

```text
calibration_confidence =
    0.25 × static_feature_inlier_score
  + 0.20 × camera_motion_fit_score
  + 0.20 × ground_reprojection_score
  + 0.15 × horizon_stability_score
  + 0.15 × height_prior_stability_score
  + 0.05 × evidence_support_score
```

If a component is not applicable, its weight is redistributed proportionally among the available components. If fewer than three components are available, or if either background motion fit or ground reprojection is unavailable, confidence is capped below `0.50`. Scores `≥ 0.75` are supported, `0.50–0.74` require a visible warning, and scores below `0.50` fail automatic calibration and require a reviewed override or simplified rendering.

The component values, normalized residuals, weights, and final score must be written to both `reconstruction_plan_v2.json` and `render_report.json`. Debug HUD mode must show calibration confidence and warn when it falls below the review threshold.

### Fallback

If automatic calibration is unreliable, use per-video configuration:

```json
{
  "horizon_normalized_y": 0.39,
  "camera_height_meters": 2.0,
  "field_of_view_degrees": 54.0,
  "ground_near_y": 0.92,
  "ground_far_y": 0.44
}
```

The first judge-facing video should be calibrated and visually reviewed manually once. The resulting configuration remains reusable only while the camera-motion classification and calibration version remain compatible.

## 10. Procedural Human System

### Mesh and rig

The first version will generate a clean low-poly human from Blender primitives. Each person receives an armature with:

- root
- pelvis
- spine
- chest
- neck
- head
- left/right upper arm
- left/right lower arm
- left/right upper leg
- left/right lower leg
- left/right foot

Body proportions will vary deterministically by track ID so the crowd does not look cloned.

### Appearance

Visible crops will be sampled to estimate:

- upper-body color
- lower-body color
- light/dark skin-range material where evidence is sufficient
- carried bag color
- vehicle body color

The system will not attempt biometric identity reconstruction or detailed faces. Faces should remain neutral and non-identifying.

### Global identity registry

Appearance and body construction must be generated once per global track ID, never once per gap. After scene intelligence completes, the pipeline writes:

```text
outputs/_work/<video>/entity_registry.json
```

Each registry entry contains:

- deterministic seed derived from video ID and global track ID
- body proportions
- upper/lower clothing colors aggregated from visible observations
- carried-object appearance
- material parameters
- preferred animation phase
- evidence count and appearance confidence
- registry schema and generator version

Every gap references `identity_registry_id`. Blender must reuse the same registry entry and cached asset in every gap. Registry invalidation occurs only when the underlying visible evidence, tracking result, or generator version changes.

### Animation states

Minimum states:

- stationary idle
- walking
- fast walking/running
- entering frame
- leaving frame

Walking will use a procedural cyclic animation:

- opposing leg swing
- opposing arm swing
- knee bend
- slight pelvis rotation
- vertical body motion
- foot contact timing

Animation speed will be derived from world-space path speed. A deterministic phase offset prevents every person from walking in synchronization.

### Lifecycle rules

- continuous: visible before and after; animate through the entire gap
- exits: visible only before; animate toward an exit and fade only at the boundary of uncertainty
- enters: visible only after; enter from a plausible boundary
- uncertain: show as lower-confidence proxy with an uncertainty halo

No person should be forced into a gap merely because they appear elsewhere in the video.

Continuity requires valid boundary evidence, not merely an observation somewhere before or after the gap. A pre-gap entity is boundary-visible only when:

- its final accepted detection lies within the configured boundary window
- its box is not substantially truncated by the image edge
- it is not marked heavily occluded
- its detection confidence passes the boundary threshold

If forward prediction places the entity outside the camera frustum before or during the gap, its lifecycle becomes `exits`; it must not remain visible for the entire interval.

### Crossing-track disambiguation

Cross-gap matching is one-to-one and uses a cost composed of appearance distance, trajectory consistency, scale consistency, and lifecycle feasibility. Appearance conflict is a hard rejection condition: two tracks must never be merged merely because their positions cross or their predicted paths are spatially close.

For crossing candidates:

1. Compare global appearance signatures aggregated across visible observations.
2. Predict both tracks independently through the gap.
3. Reject assignments that require implausible speed, heading reversal, or scale change.
4. Solve the remaining one-to-one assignment globally for that boundary.
5. Keep unmatched exit and entry tracks separate when evidence remains ambiguous.

The matcher must prefer two honest low-confidence lifecycle events over one polished but unsupported identity swap.

## 11. Vehicle System

Procedural vehicle categories:

- car
- bus
- truck
- bicycle
- motorcycle

Initial cars, buses, and trucks will use simple clean geometry with class-specific proportions. Vehicles will:

- orient along the tangent of their path
- accelerate and decelerate smoothly
- rotate wheels according to distance traveled
- maintain ground contact
- cast coherent shadows

Bicycles and motorcycles may use simplified rider proxies in the first version.

Vehicle position and heading must be generated from one C1-continuous curve. Orientation follows the curve tangent, not independently interpolated angle keyframes. Turning vehicles must respect configured maximum curvature and angular velocity. If boundary evidence cannot support a continuous turn, the vehicle receives lower fidelity or separate exit/entry lifecycles rather than an instantaneous rotation.

## 12. Environment System

### Forensic scene

The default missing-gap environment should contain:

- matte ground plane
- horizon grid or restrained lane markers
- simple proxy blocks for major static scene masses
- backplate or evidence thumbnail for context
- reconstructed entities
- optional path trails
- timecode and confidence HUD

This avoids the existing failure where a supposedly realistic background contains ghost pedestrians.

### Scene occlusion

Version 1 supports occlusion between reconstructed entities through true 3D depth.

Occlusion behind real buildings, street furniture, and other static surfaces requires one of:

- manually configured proxy geometry
- monocular depth estimation
- semantic scene masks

Proxy geometry is recommended for the first judge-facing video because it is predictable and does not require another model.

### Proxy-geometry authoring workflow

Proxy placement must be evidence-aligned rather than estimated in an empty viewport. A Blender-side authoring tool will:

1. Load the reviewed camera calibration.
2. Set Blender to the calibrated camera view.
3. Display a selected visible evidence frame as the camera backplate.
4. Allow the user to place ground polygons and simple boxes while viewing their projection over the evidence.
5. Assign proxy roles such as `ground`, `building`, `wall`, `street_furniture`, and `occluder`.
6. Save geometry and camera references to `config/proxy_geometry/<video_id>.json`.
7. Render a wireframe validation overlay for review.

Validation records screen-space corner error, ground-contact alignment, and calibration version. Proxy files must never be authored against hidden frames.

## 13. Materials, Lighting, and Shadows

Recommended look:

- physically based but restrained materials
- one sun/area key light
- soft ambient fill
- contact shadows under people and vehicles
- subtle atmospheric depth
- high-contrast evidence colors only for paths and labels

Lighting should support readability rather than cinematic drama. Confidence color convention:

- green: confidence at least 0.75
- amber: confidence 0.50–0.74
- red: confidence below 0.50

### Confidence-to-fidelity tiers

Confidence controls geometry and animation fidelity, not only color:

- `≥ 0.75 — supported`: full rig, full animation, solid material, normal path corridor
- `0.50–0.74 — plausible`: full rig and animation, reduced material opacity, visible uncertainty corridor, amber outline
- `< 0.50 — weak`: simplified silhouette/proxy, no detailed action claim, broad uncertainty corridor, red outline

A weak entity remains visible as evidence of a possible presence, but its rendering must not visually imply detailed knowledge the system does not possess. Calibration confidence can downgrade every entity in a gap by one fidelity tier when scene scale or camera pose is unreliable.

### Crowded-scene relevance score

Entity selection must be deterministic and documented:

```text
relevance = entity_confidence × proximity_factor × screen_size_factor × lifecycle_factor
```

Where:

- `entity_confidence` is the validated gap confidence in `[0, 1]`
- `proximity_factor` increases smoothly from `0.50` at the far plane to `1.00` at the near plane
- `screen_size_factor` is normalized projected area, clamped to `[0.25, 1.00]`
- `lifecycle_factor` is `1.00` for continuous, `0.75` for entering/exiting, and `0.50` for uncertain

Entities below the configured minimum relevance remain in the machine-readable report but are not promoted to detailed presentation geometry. After relevance sorting, a projected-overlap/readability budget prevents the HUD and scene from becoming visually saturated. Any emergency performance cap must be explicit in the render report; it must not silently act as the primary selection rule.

## 14. Transitions

The visible-evidence and reconstruction styles are intentionally different. The transition should explain that change instead of trying to hide it.

Proposed transition:

1. Last evidence frame freezes for 3–4 frames.
2. Detection boxes simplify into track markers.
3. A short 4–6 frame forensic-mode transition introduces the 3D camera.
4. The 3D gap plays at normal speed.
5. The scene transitions back to the first visible evidence frame.

The transition must not shorten the timeline. Transition frames belong inside the hidden interval.

## 15. HUD and Evidence Communication

Required information:

- `AI-INFERRED FORENSIC RECONSTRUCTION`
- `NOT GROUND TRUTH`
- gap number
- source frame/timecode
- overall confidence
- entity IDs when enabled
- optional legend for uncertainty colors

The HUD should be legible at 720p without covering important action. Debug paths and labels must be configurable separately from presentation mode.

## 16. Rendering Pipeline

### Blender invocation

The normal Python process will invoke Blender in background mode:

```text
blender --background --python blender/render_gap.py -- \
  --plan <reconstruction_plan_v2.json> \
  --output <gap_render_directory>
```

### Render format

Recommended intermediate format:

- PNG frame sequence
- original video resolution
- original frame rate
- deterministic frame numbering

PNG sequences are restartable and avoid losing an entire render if a process stops. After rendering, frames will be encoded into a gap video.

### Render engine

Start with Blender Eevee for speed. If this machine cannot render Eevee reliably, use Blender Workbench for the first forensic version. Cycles is not appropriate for the initial full-video workflow on this hardware.

### Cache policy

Each gap receives a cache signature based on:

- reconstruction plan hash
- Blender script version
- Blender configuration
- resolution and frame rate

Unchanged gaps should not be rerendered.

## 17. Encoding and Audio

The existing OpenCV stitcher drops audio and produces inefficient `mp4v` output.

The proposed pipeline will use FFmpeg after explicit approval to:

- encode H.264 video
- preserve the original frame rate
- concatenate visible and reconstructed segments without changing duration
- copy or remux source audio when present
- produce a broadly compatible MP4

FFmpeg is not currently installed. It is a separate dependency and must not be installed until this plan is approved.

## 18. Proposed Files

```text
config/
  blender_render_config.json
  video_calibration.json
  proxy_geometry/
    <video_id>.json

src/
  application/
    reconstruction_pipeline.py
    processing_jobs.py
  domain/
    configuration.py
    processing_job.py
    video_upload.py
  interfaces/
    http/
      local_server.py
  blender_export.py
  blender_runner.py
  camera_calibration.py
  appearance.py
  identity_registry.py
  path_prediction.py
  proxy_geometry.py
  media.py

blender/
  render_gap.py
  scene_builder.py
  camera_builder.py
  environment_builder.py
  human_builder.py
  human_animation.py
  vehicle_builder.py
  vehicle_animation.py
  materials.py
  hud.py
  author_proxies.py

tests/
  unit/
    application/
      test_processing_jobs.py
    domain/
      test_configuration.py
      test_video_upload.py
    interfaces/
      test_local_server.py
  test_blender_export.py
  test_camera_calibration.py
  test_appearance.py
  test_identity_registry.py
  test_path_prediction.py
  test_proxy_geometry.py
  test_blender_runner.py

web/
  index.html
  assets/
    styles/
      app.css
    scripts/
      api-client.js
      formatters.js
      app.js

app.py
```

Existing OpenCV compositing remains available as `--renderer 2d` until Blender output passes acceptance gates.

## 19. Proposed CLI

```bash
python app.py
python app.py --host 127.0.0.1 --port 8000
```

The browser UI is the supported operator surface. The reconstruction orchestrator lives in `src/application/reconstruction_pipeline.py`; the obsolete root `run.py` entrypoint is removed after its behavior is preserved. Preview-gap and renderer selection will be exposed through UI controls when the Blender phases implement them.

## 20. Implementation Phases and Approval Gates

### Phase 0 — Freeze and benchmark the baseline (implemented)

Work:

- preserve current 2.5D renderer as fallback
- record current output metrics and sample frames
- define a fixed seed and fixed first preview gap
- add renderer selection to configuration, without changing default behavior
- verify that Blender background mode can consume a system-Python-generated JSON fixture without importing `src/`
- benchmark one representative frame and one short sequence in Eevee and Workbench; record seconds per frame, memory use, and visible quality defects before choosing the default engine

Gate:

- existing six tests still pass
- current output remains reproducible
- Blender starts in background mode, respects the JSON boundary, and completes a low-resolution render
- the selected render engine and benchmark evidence are recorded in the render configuration

### Phase 1 — Plan v2 and coordinate conversion (implemented)

Work:

- define strict plan-v2 validation
- infer entity lifecycle per gap, including strict boundary visibility and predicted pre-gap exits
- build the global per-track identity registry and aggregate appearance evidence once per video
- convert image observations into at least three-point, forward-predicted world paths using the declared curve model
- add crossing-track disambiguation and heading-disagreement confidence rules
- add static-versus-dynamic camera classification, robust height priors, and separate calibration confidence
- record post-gap position and heading residuals without forcing the prediction to meet them

Gate:

- plan fixtures validate
- invalid inputs fail cleanly
- world paths preserve left/right, near/far, and relative scale
- every renderable path has at least three waypoints and names its interpolation method
- the same track resolves to the same identity-registry record in every gap
- calibration confidence, component residuals, heading disagreement, and post-gap residuals are present and testable
- an appearance-signature conflict prevents a crossing-track merge even when the spatial match is plausible

### Phase 2 — Single-frame Blender scene (implemented and visually reviewed)

Work:

- build camera, ground plane, lighting, HUD, and proxy environment
- author proxy geometry through the calibrated evidence camera with the visible backplate overlaid
- save the proxy contract and validate its wireframe projection against visible evidence
- build procedural person and vehicle meshes
- render a single midpoint frame for one gap

Gate:

- user reviews one rendered frame
- perspective, scale, color, labeling, and style are approved
- calibration confidence and its warning state are visible in debug HUD mode
- proxy wireframes align with evidence geometry within the configured screen-space tolerance

No animation work should proceed before this visual gate.

### Phase 3 — One-gap animation (implemented; final user approval gate)

Work:

- add human armatures and walking cycles
- add vehicle orientation and wheel animation driven by a C1-continuous path tangent
- add entity lifecycles and uncertainty paths
- apply confidence-to-fidelity tiers and the formal crowded-scene relevance score
- render only gap 0

Gate:

- no sliding static humans
- no ghosting or double exposure
- no floating feet
- entities move in the correct direction
- low-confidence entities use simplified silhouettes rather than unsupported detailed action
- vehicle heading changes continuously and respects curvature limits
- gap duration and frame count are exact
- user approves the one-gap video

### Phase 4 — Three representative gaps (pending)

Work:

- render an easy gap, a crowded gap, and a low-confidence gap
- tune camera consistency and performance
- test cache invalidation and restartability

Gate:

- all three gaps meet the visual checklist
- crowded scenes remain readable
- uncertainty is visible but not distracting
- repeated global tracks retain identical cached appearance and proportions across the three renders
- every excluded crowded-scene entity remains listed with its relevance score and exclusion reason

### Phase 5 — Full one-video render (pending)

Work:

- render all gaps for `input_vid3.mp4`
- encode and stitch the full timeline
- preserve audio
- run evaluation

Gate:

- exact source duration, resolution, and frame rate
- no missing or duplicate timeline frames
- no failed gap renders
- output passes automated checks and visual review

### Phase 6 — Judge-facing polish (partially implemented; final gate pending)

Work:

- presentation HUD mode
- opening legend explaining evidence versus inference
- concise end card with confidence and limitations
- produce a short curated demo excerpt in addition to the full result
- update README and architecture documentation

Gate:

- someone unfamiliar with the project can distinguish evidence from inference
- no claim implies recovered ground truth
- user approves the final judge-facing video

### Phase 7 — Second video (pending)

Only after `input_vid3.mp4` is approved:

- calibrate the second camera
- run the same fixed gates
- render `input_vid4.mp4`

## 21. Acceptance Criteria

### Visual

- no translucent duplicate people
- no background cross-dissolve ghosts
- people visibly walk, idle, enter, or exit
- vehicles face their direction of travel
- feet and wheels remain on the ground
- entity scale is plausible across depth
- shadows are stable
- camera does not jump inside a gap
- HUD is legible and professional

### Evidence correctness

- hidden truth is not read before evaluation
- only boundary-supported entities enter the gap plan
- continuous identities have valid, unoccluded boundary evidence and do not merely appear somewhere in the visible range
- entering/exiting identities are labeled with lower confidence
- appearance colors come only from visible evidence
- uncertainty increases with weaker continuity
- post-gap observations are soft consistency checks and never hard path or speed targets
- conflicting appearance signatures are never merged during crossing-track matching
- the same global track identity retains its proportions, appearance, materials, carried items, and animation phase across all gaps
- calibration confidence remains separate from entity confidence and can downgrade visual fidelity for the whole gap

### Technical

- exact original frame count
- exact original frame rate within container precision
- original resolution
- source audio preserved when present
- deterministic output for a fixed seed and configuration
- resumable per-gap renders
- Blender failure returns a clean error and preserves logs
- no new unapproved external asset dependencies

### Evaluation

Existing metrics remain, but they are not sufficient. Add:

- lifecycle consistency
- direction agreement
- world-path continuity
- entry/exit boundary position error
- post-gap position and heading residuals before any presentation smoothing
- heading-disagreement and identity-ambiguity rates
- calibration confidence, reprojection error, and height-prior stability
- cross-gap identity appearance and proportion consistency
- crowded-scene selection stability and excluded-entity reporting
- animation state consistency
- render completion and dropped-frame checks

## 22. Risks and Mitigations

### Incorrect identities across a gap

Risk: the renderer creates a polished animation for the wrong person.

Mitigation:

- stricter appearance and spatial matching
- reject ambiguous matches
- show separate exit/entry entities rather than forcing continuity
- lower confidence visibly

### Camera calibration is inaccurate

Risk: figures float or scale incorrectly.

Mitigation:

- per-video calibration file
- single-frame visual approval gate
- explicit static-versus-dynamic camera classification
- per-frame stabilization or camera motion tracks when the camera is not static; never reuse one global homography across moving footage
- median same-track height priors with boundary/occlusion filtering, MAD outlier rejection, minimum observation counts, and variance warnings
- separate calibration confidence and component residuals in the plan, debug HUD, and render report
- manual override for the judge-facing videos

### Crowded scenes

Risk: too many entities make the scene unreadable.

Mitigation:

- calculate `relevance = entity_confidence × proximity_factor × screen_size_factor × lifecycle_factor`
- normalize proximity to `0.50–1.00`, screen-space size to `0.25–1.00`, and use lifecycle factors `1.00` for continuous, `0.75` for enter/exit, and `0.50` for uncertain entities
- apply explicit lifecycle filtering and keep below-threshold entities in the report even when they are not rendered
- use projected screen overlap and a configurable readability budget to select among otherwise relevant entities
- permit an emergency presentation cap only when its configured value, excluded IDs, scores, and reasons are written to the render report

### Cross-gap identity inconsistency

Risk: the same tracked person or vehicle changes proportions, colors, carried objects, or animation style between gaps.

Mitigation:

- create one deterministic identity registry per video before any Blender export
- derive the seed from video identity and global track ID, never gap index
- aggregate visible appearance evidence once and cache the generated asset parameters
- reference the same registry record from every gap and test that regenerated plans are byte-stable for a fixed input and seed

### Procedural humans look too simple

Risk: output resembles a basic game prototype.

Mitigation:

- coherent forensic art direction
- varied proportions and colors
- high-quality walk cycles
- professional lighting and HUD
- avoid pretending to be photorealistic

### Rendering is slow

Risk: CPU-only full rendering takes too long.

Mitigation:

- Eevee or Workbench
- render one preview gap first
- PNG caching
- reuse unchanged environment assets
- configurable preview resolution and samples

### Blender scripting compatibility

Risk: `bpy` APIs change.

Mitigation:

- pin Blender 4.5 LTS
- record Blender version in every render report
- test script startup and one-frame render in CI/local checks

### Additional dependencies

Risk: the project becomes hard to reproduce.

Mitigation:

- procedural assets first
- no third-party asset packs in version 1
- explicit Blender and FFmpeg preflight checks
- documented setup command

## 23. Testing Strategy

### Normal Python tests

- plan-v2 validation
- coordinate mapping
- lifecycle inference, including invalid boundary observations and predicted off-screen exits
- appearance-color extraction
- robust median/MAD height-prior estimation and unstable-track rejection
- static-versus-dynamic camera classification and calibration-confidence scoring
- minimum-three-waypoint Catmull-Rom path construction
- post-gap residual calculation without forced arrival
- heading-disagreement thresholds and confidence penalties
- crossing-track assignment with appearance-conflict hard rejection
- global identity-registry determinism and cross-gap reuse
- confidence-to-fidelity tier selection and calibration-based downgrades
- crowded-scene relevance formula and readability-budget selection
- proxy-geometry schema validation and screen-space projection checks
- Blender command construction
- cache signature behavior
- FFmpeg command construction

### Blender smoke tests

- start Blender in background mode
- import every Blender module
- build an empty forensic scene
- create one person and one vehicle
- render one low-resolution frame
- verify the frame exists and is not blank

### Visual regression fixtures

For a fixed plan:

- save approved midpoint render
- save approved one-gap contact sheet
- compare dimensions and coarse image statistics automatically
- require manual approval for meaningful style changes

### End-to-end test

Use a short fixture video with:

- one moving person
- one stationary person
- one vehicle
- one entering entity
- one exiting entity
- two visually distinct people crossing in opposite directions
- one turning vehicle

Verify complete selection, plan export, Blender render, stitch, audio, and evaluation. The crossing identities must remain separate, heading conflicts and post-gap residuals must appear in evaluation, the turning vehicle must retain C1-continuous position and heading, and repeated global track IDs must render with identical cached identity parameters.

## 24. Logging and Failure Handling

Every Blender gap render should write:

```text
outputs/_work/<video>/gaps/gap_XX/blender/
  plan_v2.json
  calibration_report.json
  render_config.json
  proxy_validation.json
  blender.log
  scene.blend
  frames/
  gap_blender.mp4
  render_report.json
```

`render_report.json` should include:

- Blender version
- render engine
- start/end time
- frame count
- resolution
- plan hash
- script version
- calibration confidence, its component residuals, and any fidelity downgrade it caused
- post-gap position and heading residuals
- entity counts by confidence-to-fidelity tier
- relevance scores, readability exclusions, and any emergency-cap decisions
- identity-registry schema and generator versions
- warnings
- failure reason when applicable

Failures should stop the final stitch unless the user explicitly enables fallback rendering.

## 25. Definition of Done

The Blender phase is complete only when:

1. One full input video renders from a single command.
2. All selected gaps use Blender forensic scenes.
3. People have visible articulated animation.
4. Vehicles have coherent orientation and motion.
5. No 2.5D ghost compositing appears in Blender mode.
6. All hidden gaps total approximately 25% and individually remain 1–3 seconds.
7. Hidden truth is used only after reconstruction.
8. Final frame count, resolution, frame rate, and audio match the source contract.
9. Automated tests pass.
10. The user approves the one-gap, three-gap, and full-video visual gates.
11. README accurately describes what is implemented.
12. The result can be presented to judges as an intentional forensic reconstruction rather than a photorealistic recovery claim.
13. Moving-camera footage uses a validated motion track or stabilization model rather than one global homography.
14. Detailed rigs and actions appear only when entity and calibration confidence support that fidelity tier.
15. Every repeated global track keeps the same cached appearance, proportions, materials, carried items, and animation style across gaps.
16. Predicted paths retain honest post-gap residuals and are never forced to meet future observations.

## 26. Approved Implementation Decisions

The following decisions are approved for implementation and remain change-controlled:

1. **Visual style:** use deliberate stylized forensic 3D as the default.
2. **Renderer:** Blender 4.5 LTS through checked-in `bpy` scripts, not Blender MCP.
3. **Assets:** procedural humans and vehicles first; no downloaded asset packs.
4. **Calibration:** allow one reviewed per-video calibration file for judge-facing inputs.
5. **Review sequence:** one frame, then one gap, then three gaps, then one full video.
6. **FFmpeg:** install only after plan approval to preserve audio and improve encoding.
7. **Fallback:** keep the current 2.5D renderer available but never silently substitute it in a Blender judge render.
8. **Claims:** call the output an AI-inferred forensic reconstruction, not recovered footage.
9. **Prediction:** use pre-gap motion as the primary path estimate and post-gap observations only as soft consistency checks.
10. **Identity:** generate and cache appearance and body parameters once per global track ID, then reuse them across every gap.

## 27. Local Processing UI

### Goal

Provide one clean local interface that a judge or operator can use without terminal commands. Processing time is allowed, but the interface must continuously communicate what the system is doing and must never appear frozen.

### Input workflow

- support browsing and drag-and-drop upload
- accept common video containers: MP4, MOV, AVI, MKV, M4V, WebM, MPEG/MPG, and WMV
- sanitize filenames, reject unsupported extensions, enforce a configured upload-size limit, and validate that OpenCV can read the uploaded video before queueing expensive work
- store each upload and its output in a unique job directory so repeated filenames cannot overwrite one another
- process one reconstruction job at a time by default because detection and rendering are CPU/GPU intensive

### Processing experience

- show queued, validating, selecting gaps, detecting/tracking, planning, rendering, evaluating, stitching, cancelling, cancelled, completed, and failed stages
- expose real stage progress from the Python pipeline rather than a purely decorative percentage
- show elapsed time and a clearly labeled best-effort ETA derived from observed progress; count it down between pipeline updates and switch to `recalculating` when an in-flight task exceeds the latest estimate
- provide an expandable live-activity panel per job with completed, active, and pending stages, per-stage percentage, timestamps, and recent persisted pipeline messages
- stream Blender frame markers from every active worker into aggregate render progress so a long individual gap does not leave the percentage visually frozen
- open the live-activity panel automatically for a newly submitted reconstruction
- poll local job state without reloading the page and preserve completed job metadata across server restarts
- surface clean errors in the UI while retaining detailed local logs
- expose a cancel action for queued and processing jobs; show `cancelling` until active Blender/FFmpeg processes have stopped
- keep one top-level video job active at a time while rendering up to three independent Blender gaps concurrently through bounded worker threads and separate Blender subprocesses

### Page layout

- keep the upload workflow as the full-width horizontal first step on desktop and stack it cleanly on narrow screens
- present the 1–3 second gap, approximately 25% inference, three-worker rendering, and local-processing contract as compact inline facts
- omit non-functional decorative engine panels so screen space is reserved for upload controls, live progress, and outputs
- use restrained type sizes, consistent spacing, and the supplied dark/cyan visual language in both light and dark themes

### Output gallery

- automatically list every completed reconstruction
- provide in-browser playback, download, filename, size, completion time, and processing duration
- allow deletion only after explicit confirmation
- deleting an output must remove its UI record, rendered video, per-job work directory, and retained upload from disk
- reject deletion while a job is processing and validate every deletion target remains inside the managed upload/output roots

### Local API

```text
GET    /api/health
GET    /api/jobs
POST   /api/jobs
GET    /api/jobs/<job_id>
POST   /api/jobs/<job_id>/cancel
GET    /api/outputs/<job_id>
DELETE /api/jobs/<job_id>
```

Video responses must support HTTP range requests so browser seeking works. The UI and API bind to `127.0.0.1` by default and add no network service or frontend framework dependency.

### UI acceptance gate

- a supported test video can be uploaded without using the terminal
- progress, current stage, elapsed time, and ETA update while processing
- the ETA continues changing between stage updates and clearly reports when it is recalculating
- the live-activity dropdown identifies completed, active, and pending work and preserves its open state across polling refreshes
- cancellation stops active reconstruction processes and reaches a terminal cancelled state without deleting unrelated outputs
- a completed output appears automatically and plays in the browser
- download returns the physical output file
- confirmed deletion removes the record and all job-owned files from disk
- an unsupported upload fails cleanly without starting reconstruction
- the layout remains usable on desktop and mobile widths
- the top-right dark/light theme toggle follows system preference initially and remembers the operator's explicit choice

## 28. Google Colab Batch Execution

The cloud path uses one checked-in `colab/reconstruction.ipynb` as its operator interface. The notebook clones the current `main` branch and imports the existing application pipeline; reconstruction, evidence validation, Blender scene construction, evaluation, and stitching remain common code rather than notebook copies.

The notebook must:

- refuse to start expensive work when Colab has not assigned an NVIDIA GPU
- install the pinned Blender 4.5 LTS binary, project Python requirements, and FFmpeg inside the ephemeral runtime
- process from `/content` instead of directly against mounted Drive
- accept one validated common-format video and derive a content-based run identifier
- show live stage, detail, and aggregate progress emitted by the shared pipeline
- default to two Blender gap workers on a single Colab GPU and expose a one-worker fallback for memory pressure
- copy only complete Blender gap artifacts to Google Drive from a background checkpoint watcher
- restore compatible completed-gap checkpoints when the same video and deterministic seed are rerun
- save the final video and JSON reports to Google Drive before offering preview and download
- clearly state that Colab resources and runtime duration are not guaranteed

The notebook does not start `app.py`, expose a tunnel, or replace the local judge-facing browser UI. Its purpose is batch GPU execution with durable outputs while preserving the same evidence and rendering contracts.
