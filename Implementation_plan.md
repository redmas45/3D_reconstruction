# Blender Forensic Reconstruction — Implementation Plan

## 1. Document Purpose

This is the approved, living implementation plan for the AI evidence-gap reconstruction project. It records the target architecture, acceptance gates, completed work, measured results, and remaining validation work.

### Implementation status — audited 23 July 2026

- Phase 0 is implemented: the OpenCV 2.5D renderer remains an explicit fallback and Blender 4.5 LTS runs headlessly through a JSON-only boundary.
- Phase 1 is substantially implemented and unit tested: visible-only evidence enforcement, delayed hidden-truth materialization, identity registry, motion measurement, robust height priors, forward-predicted three-point paths, soft post-gap residuals, crossing-appearance rejection, and confidence-driven presentation filtering.
- Phase 2 is **partial**, not complete: source resolution, visible-person ground-contact/horizon fitting, ground grid, evidence inset, procedural entities, confidence HUD, and neutral/vehicle-supported proxy profiles exist. Full focal-length/pose calibration, proxy authoring, wireframe reprojection validation, and source-camera motion application do not.
- Phase 3 is **partial**: primitive-based articulated motion, lifecycle fades, confidence fidelity tiers, transition shutters, fractional frame rates, and source resolution are implemented. Dynamic footage is explicitly downgraded to a labeled stabilized forensic view. Human armatures/skeletal rigs remain optional future polish rather than a current acceptance dependency.
- UI/pipeline integration is implemented, including one-video queueing, three local Blender gap workers, cancellation, live progress, output playback/deletion, and a 2.5D fallback. Metadata writes are throttled and transactional so frame callbacks do not create Windows write contention or ghost jobs.
- Runtime hardening now performs FFmpeg, FFprobe, and Blender preflight; rejects variable-frame-rate, odd-dimension, over-4K-budget, over-120-fps, and over-10-minute sources; validates decoded segment/output contracts; preserves portrait and landscape dimensions; stops sibling Blender processes after one gap fails; and prevents short source audio from truncating video.
- Resume state is content-addressed by source SHA-256. Selection and detection JSON use atomic replacement and malformed-cache recovery; Blender reuse verifies the plan/report contract and the physical MP4 contract before accepting a completed gap.
- The local service is single-instance and loopback-only, validates request hosts, bounds request/upload stalls, persists clean public errors, supports cancellation, and closes active browser connections during shutdown.
- The supported profile is currently constant-frame-rate, static-camera footage with people or road vehicles visible near gap boundaries. Moving-camera footage is experimental because motion is measured but not applied to the Blender camera; arbitrary indoor/general-scene reconstruction is not claimed as flawless.
- Evidence/story Gates A–C are implemented for Blender jobs: deterministic clue IDs, a hidden-frame-safe keyframe/crop manifest, per-entity hypotheses, batched multimodal Azure gap decisions, strict local validation, deterministic fallback, a presentation-only whole-video narrative, renderer-only storyboard compilation, render budgets, and the UI story/gap timeline are present. The configured deployment is read from `AZURE_OPENAI_CHAT_DEPLOYMENT`; a live paid end-to-end request and visual comparison remain validation gates.
- The Colab T4 successfully completed a real Blender Cycles OptiX probe in 4.544 seconds, proving GPU access. A subsequent production run reached the Colab T4 session limit after roughly 3–4 hours, proving that the current source-FPS, 75%-scale, 16-sample profile is not a viable full-video default.
- Smart-renderer Gate D is substantially implemented: the default profile renders an 8 fps, 45%-scale, two-sample composite sequence (Colab: 6 fps at 40%) with atomic SHA-256 frame manifests, skips valid frames after interruption, and restores exact source timing. Bounded environment, actor, uncertainty, HUD, depth, and shadow diagnostics are extracted at review poses from the same render. The heaviest gap runs first, predicts total time, writes `runtime_estimate.json`, refuses projections over 45 minutes without override, and in Colab requires explicit visual approval before remaining gaps start. Reusable scene-shell instancing and full-frame layer manifests remain future optimization.
- Phase 4 (three representative gaps), Phase 5 (full one-video render), Phase 6 judge polish, and Phase 7 second-video validation remain approval gates.

Historical render artifacts (useful for timing/container inspection, but generated before the current evidence/calibration hardening and therefore not current visual-acceptance proof):

- `outputs/blender_preview_input_vid3/gap_00/`: original-video gap-0 midpoint, animation, plan, `.blend`, report, log, and contact sheet.
- `outputs/e2e_blender_smoke/`: 150-frame UI-pipeline smoke result with one 38-frame Blender gap, H.264 video, AAC audio, and exact 1280×720 / 29.970029 fps contract.
- Eevee production timing on this CPU: final 57-frame gap with transitions in 560.489 seconds; 38-frame smoke gap in 261.256 seconds.
- Automated verification: 126 tests pass. Coverage includes source admission, content-addressed and corrupt-cache recovery, hidden-frame image isolation, per-entity decision validation, narrative/storyboard compilation, Azure multimodal request construction, sparse timing normalization, Story v2 public artifacts, sibling cancellation, metadata rollback, audio duration, local API races, UI-controller behavior, and Cycles GPU configuration.
- A real headless Blender contract probe preserved a 720×1280 portrait resolution at 29.97 fps. A new full render has not been run during this audit.

## 2. Current State

The project now:

- accepts one uploaded video at a time through the local UI or Colab notebook
- distributes approximately 25% of the video across random 1–3 second gaps
- keeps the remaining 75% as YOLO-annotated evidence
- tracks visible people, vehicles, and carried objects
- writes the first bounded evidence ledger and can call the configured Azure OpenAI deployment for strict hypothesis selection
- writes strict plan-v2, identity-registry, camera-motion, selection, and render-report contracts
- renders inferred gaps with headless Blender forensic 3D by default
- retains OpenCV 2.5D only as an operator-selected fallback
- prevents hidden-frame detections and does not materialize hidden truth until every inferred gap is rendered
- evaluates completed gaps only after rendering
- stitches the exact original frame count and uses FFmpeg to preserve source audio

The earlier OpenCV output had background dissolves, duplicated pedestrians, sliding cutouts, weak masks, and no scene geometry. That output remains available only as a compatibility fallback. The current Blender direction uses deliberate stylized forensic 3D and exposes uncertainty rather than attempting an untrustworthy photoreal recovery.

Blender 4.5.10 LTS is installed and the checked-in integration is available. FFmpeg integration is implemented, but FFmpeg is **not currently discoverable on this workstation**; the pipeline now stops at preflight before expensive detection/rendering until it is installed. Blender reads validated JSON contracts and does not import the system-Python business modules.

## 2.1 Authoritative Next Build: Evidence Narrative and Smart Rendering

This section is the controlling plan for the next implementation. If an older phase or rendering note conflicts with this section, this section wins. The project must complete the evidence-narrative, storyboard, benchmark, and resume gates before another full-video render.

### Product statement

The system uses the visible 75% of the video to build an auditable reconstruction narrative for the distributed missing 25%. It does not ask a model to freely invent a story and it does not claim to recover historical truth.

The judge-facing promise is:

> The system extracts visible clues, connects them into the most evidence-supported reconstruction narrative, converts that narrative into constrained renderable event beats, and visualizes those beats as clearly labeled forensic 3D inference.

The term **story** in the UI means an **evidence-grounded reconstruction narrative**. Every material claim must reference visible evidence, a validated entity, or a deterministic hypothesis. Unsupported actions, identities, objects, locations, and coordinates are forbidden.

Coherence must never be manufactured. If the visible evidence supports only independent movements rather than one causal event, the narrative must say so and summarize the gaps independently. “No supported global causal story” is a valid, honest result.

### Required user experience

After upload, the UI must progressively show four distinct products:

1. **Evidence clues:** concise bullet points extracted from visible frames, tracks, motion, appearances, scene context, and gap boundaries.
2. **Reconstructed story:** a short whole-video summary explaining the most supported sequence of visible and inferred events.
3. **Gap timeline:** one card per missing interval showing what was observed before, what is inferred inside, what was observed after, confidence, unknowns, and rejected alternatives.
4. **Rendered reconstruction:** a storyboard-driven 3D visualization whose entities, paths, actions, and uncertainty match the accepted structured story.

Observed and inferred information must never be visually blended into one category:

- **Observed** uses the evidence color and names the source frame/track.
- **Inferred** uses the reconstruction color and names the selected hypothesis.
- **Unknown** uses the warning color and states what the system cannot establish.
- **Rejected** remains inspectable but visually secondary.

### Terminology and authority boundaries

- **Evidence fact:** deterministic fact extracted from a visible frame or visible track.
- **Clue:** human-readable statement derived from one or more evidence facts.
- **Hypothesis:** deterministic, complete, renderable candidate for an entity during one gap.
- **Gap decision:** selected entity hypotheses plus chronological event beats for one missing interval.
- **Reconstruction narrative:** the accepted gap decisions arranged into a coherent whole-video explanation.
- **Storyboard:** renderer-only contract compiled deterministically from accepted gap decisions.
- **Presentation summary:** readable prose shown in the UI; it cannot introduce renderer instructions.

Authority is intentionally split:

- YOLO/tracking/camera modules decide what was visibly detected and measured.
- Normal Python creates IDs, coordinates, paths, physical constraints, and candidate hypotheses.
- Azure OpenAI compares supplied evidence and chooses or rejects bounded hypotheses.
- Normal Python validates the decision and compiles the storyboard.
- Blender renders the storyboard but makes no inference decisions.
- The prose summary explains accepted decisions but cannot change them.

### End-to-end flow

```text
Uploaded video
    |
    +-- validate source timing, resolution, duration, and tools
    |
    +-- select distributed 1–3 second gaps totaling approximately 25%
    |
    +-- analyze only the visible 75%
    |       |
    |       +-- detections and sequential tracking
    |       +-- global identity registry
    |       +-- camera/scene measurements
    |       +-- visible boundary observations
    |       +-- visible keyframe and entity-crop evidence pack
    |
    +-- deterministic clue catalog
    |
    +-- deterministic per-entity hypothesis library
    |
    +-- Azure gap planner
    |       |
    |       +-- selects supplied hypotheses
    |       +-- writes evidence references
    |       +-- writes chronological renderable beats
    |       +-- reports confidence, alternatives, and unknowns
    |
    +-- local decision validator
    |
    +-- deterministic storyboard compiler
    |
    +-- Azure narrative synthesizer
    |       |
    |       +-- clue bullets for presentation
    |       +-- whole-video reconstruction summary
    |       +-- per-gap readable summaries
    |
    +-- representative-gap render benchmark and visual gate
    |
    +-- resumable layered sparse-frame rendering
    |
    +-- exact frame-rate/resolution conversion and final stitch
    |
    +-- hidden-truth evaluation only after rendering
```

### Stage A — visible evidence pack

The evidence stage scans all visible intervals, not only the immediate gap boundaries. This is how the project learns the scene and continuity from the visible 75%.

It produces stable evidence facts for:

- global scene type and camera-motion classification
- visible people, vehicles, carried objects, and relevant static objects
- stable track IDs and cross-gap identity associations
- entity appearance colors and proportions from visible samples
- motion direction, velocity, acceleration estimate, and heading consistency
- lifecycle state: continuous, enters, exits, or uncertain
- scene entrances, exits, walkable/driveable regions, and occluding proxies
- pre-gap and post-gap boundary observations with frame references
- conflicts such as appearance mismatch, heading disagreement, track crossing, or unstable calibration
- explicit missing information

The visual evidence pack may include only visible images:

- a bounded set of global scene keyframes selected by scene change and track coverage
- up to two pre-gap and two post-gap boundary frames per gap
- small visible entity crops used to distinguish identities and appearance
- contact sheets that label every image with its evidence ID and source frame

No hidden frame may be exported into this pack. Every image path and frame index must pass the same hidden-range validator as the structured ledger. Images are resized and use an economical detail level by default; high/original detail is reserved for an explicitly justified spatial ambiguity. The Responses API supports multiple image inputs, but image inputs consume tokens, so the evidence pack must remain deliberately bounded. See the official [OpenAI images and vision guide](https://developers.openai.com/api/docs/guides/images-vision).

Vision input supplies semantic context such as scene type, relationships, appearance, and possible actions. It is never authoritative for exact counts, pixel coordinates, speed, scale, or identity; deterministic detection, tracking, and calibration remain authoritative for those measurements.

### Stage B — deterministic clue catalog

Clues must exist before the Azure call so the UI can show progress and the system still has an auditable result if Azure is unavailable.

Each clue has:

```json
{
  "clue_id": "clue_track_person_6_pre_gap_03",
  "category": "motion",
  "observation": "Person 6 moves from left to right before gap 3.",
  "evidence_references": ["track:person_6:frame:418", "track:person_6:frame:425"],
  "entity_ids": ["person_6"],
  "gap_indexes": [3],
  "confidence": 0.84,
  "conflicts": [],
  "observed": true
}
```

Clue categories are scene, identity, appearance, motion, lifecycle, boundary, camera, occlusion, conflict, and unknown. Clue text is deterministic and conservative. Azure may rephrase a clue for readability, but it may not change its meaning or evidence references.

### Stage C — per-entity hypothesis library

The current gap-wide three-choice hypothesis is too coarse because one uncertain entity should not slow or stop every other entity. The next schema moves selection to each entity inside each gap.

Candidate actions are generated only when supported by lifecycle and evidence:

- `continue_measured_motion`
- `continue_reduced_motion`
- `hold_position`
- `exit_visible_region`
- `enter_visible_region`
- `follow_supported_turn`
- `remain_occluded`
- `identity_unresolved_proxy`

Every hypothesis contains:

- stable hypothesis, gap, track, and identity IDs
- complete three-or-more-point world path
- animation state and speed range
- start, midpoint, and end visibility
- allowed action tokens
- collision/occlusion constraints
- evidence prior and calibration confidence
- visual fidelity tier
- reasons the candidate exists
- conditions that make it invalid

Azure never returns new coordinates. It selects IDs and event tokens already present in this library.

### Stage D — Azure gap planner

The first Azure pass is the only model output allowed to influence rendering. It receives the compact evidence ledger, clue catalog, entity hypothesis library, and bounded visible evidence images.

For a small video, one request may contain all gaps. When the evidence pack exceeds the configured token/image budget, gaps are processed in deterministic chronological batches while a small global scene/identity header is repeated. Batch boundaries must never split one gap.

The output uses strict Structured Outputs through `text.format`, because the response is application data rendered into separate UI and storyboard fields. OpenAI documents structured response formats as the appropriate choice when the model should return a schema-shaped response rather than call application tools. Application validation remains mandatory because schema adherence does not prove factual correctness. See the official [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

The conceptual response is:

```json
{
  "schema_version": 2,
  "evidence_digest": "...",
  "gap_decisions": [
    {
      "gap_index": 3,
      "gap_summary": "Person 6 most likely continues rightward while the vehicle remains behind.",
      "entity_decisions": [
        {
          "entity_id": "person_6",
          "selected_hypothesis_id": "gap_03_person_6_continue_measured",
          "evidence_references": ["clue_track_person_6_pre_gap_03"],
          "confidence": 0.78,
          "unknowns": ["Exact arm pose is not observable."],
          "rejected_hypotheses": [
            {
              "hypothesis_id": "gap_03_person_6_hold",
              "reason": "Visible pre-gap motion conflicts with a stationary hold."
            }
          ]
        }
      ],
      "event_beats": [
        {
          "time_fraction": 0.0,
          "entity_id": "person_6",
          "action_token": "walk",
          "path_id": "gap_03_person_6_continue_measured"
        }
      ],
      "confidence": 0.74,
      "unknowns": ["No hidden-frame observation is available."]
    }
  ]
}
```

The model must be instructed to abstain through an allowed unresolved hypothesis when evidence cannot support a more specific action. A refusal, incomplete response, invalid reference, or unsupported claim never reaches Blender.

### Stage E — decision validation and storyboard compilation

Local validation checks:

- exact evidence digest and schema version
- exactly one decision for every renderable gap entity
- known gap, entity, clue, hypothesis, path, and action IDs
- evidence references belong to the same video and permitted gap context
- all confidence values are finite and within `0–1`
- event beats are ordered and remain within `0–1` gap time
- lifecycle/action compatibility
- continuous vehicle headings and physically plausible speed/acceleration
- no hidden-frame references
- no prose field is treated as executable input

After validation, Python compiles `render_storyboard.json`. The compiler—not Azure—resolves path coordinates, Blender frame numbers, entity assets, materials, animation parameters, camera contract, occlusion layers, and uncertainty graphics.

### Stage F — whole-video narrative synthesizer

The second Azure pass is presentation-only. It reads the validated clue catalog and accepted gap decisions, not the raw hypothesis set, and produces:

- a one-sentence headline
- a concise whole-video reconstruction summary
- chronological story points
- readable per-gap summaries
- the most important supporting clues
- global confidence and unknowns

The schema includes a `causal_link_supported` flag. When false, the summary may connect events chronologically but must not claim that one caused another.

Every sentence-level story item carries evidence/clue references. The synthesizer cannot change selected hypotheses or storyboard data. If it fails, the UI uses a deterministic summary assembled from accepted decisions and rendering continues.

The UI and artifacts expose concise conclusions, references, rejected alternatives, confidence, and unknowns. Raw private chain-of-thought is never requested, stored, logged, or shown.

### Narrative UI specification

The processing card contains an expandable **Evidence and reconstructed story** panel with three tabs or stacked sections:

**Evidence clues**

- bullet list grouped by scene, entities, motion, boundaries, conflicts, and unknowns
- observed badge, confidence, and compact frame/track reference
- available progressively as deterministic extraction finishes

**Reconstructed story**

- whole-video summary at the top
- explicit prefix: “Based on visible evidence, the system infers…”
- Azure-assisted or deterministic-fallback badge
- global confidence and unknowns
- no sensational or certainty-implying language

**Gap timeline**

- gap number, source time range, and duration
- “Before — observed”, “Inside gap — inferred”, and “After — observed” columns
- selected entity actions and paths
- supporting clue IDs
- rejected alternatives
- confidence/fidelity tier
- preview status and render status

The panel remains available after completion in the output gallery. Judges must be able to understand why the animation exists without opening JSON files.

### Artifact contract

```text
evidence/
  evidence_ledger_v2.json
  clue_catalog.json
  visual_evidence_manifest.json
  keyframes/<evidence_id>.jpg
  crops/<evidence_id>.jpg

reasoning/
  gap_hypotheses_v2.json
  gap_decisions_v2.json
  reconstruction_narrative.json
  reasoning_report.json
  reasoning_cache.json

storyboard/
  render_storyboard.json
  scene_shell_manifest.json
  render_budget.json

renders/gap_XX/
  frame_manifest.json
  environment/
  actors/
  shadows/
  composite/
  gap_normalized.mp4
```

All caches include source-video hash, evidence digest, prompt/schema version, deployment, Blender version, render profile, and relevant asset hashes. A changed clue, accepted decision, scene shell, or render setting invalidates only its downstream artifacts.

### Smart rendering objective

Rendering must optimize the number of expensive scene samples rather than trying to consume all available RAM or launch competing GPU processes. The target is visually coherent stylized 3D, exact final timing, and resumability—not source-FPS path tracing.

The authoritative design is a **storyboard-first, layered, sparse-frame renderer**.

### One reusable scene shell per video

Before gap rendering, Blender builds one content-addressed `scene_shell.blend` containing:

- calibrated camera and color management
- ground plane and proxy environment
- lighting rig and world settings
- global identity-linked materials and entity assets
- render layers and compositor nodes

Each gap creates only animation and visibility overrides against the shell. Static-camera scenes reuse the environment layer across sparse frames. Moving-camera footage either renders the environment per sparse frame with the measured camera track or is downgraded to the explicit experimental/fallback path; a static shell must never be falsely reused for a moving camera.

### Layered render passes

Each gap is separated into:

1. **Environment color/depth pass:** static or slowly changing scene context plus depth/holdout data, rendered once when camera/environment are static.
2. **Actor beauty/depth pass:** transparent people and vehicles at sparse reconstruction frames.
3. **Shadow/contact pass:** dynamic ground contact and actor shadows at the same sparse frames.
4. **Uncertainty pass:** paths, halos, and proxy silhouettes where required.
5. **HUD/composite pass:** depth-aware occlusion, evidence label, gap summary, confidence, and timecode added outside the expensive Cycles beauty pass.

This preserves shadows and depth while avoiding a full expensive environment render for every output frame.

### Reconstruction frame rate

Blender renders animation at an effective **10–12 reconstruction FPS** by default. The exact value is selected as a clean divisor or near-divisor of the source rate so final frame mapping is deterministic:

- approximately 10 fps for 29.97/30 fps sources
- 12 fps for 24/48/60 fps sources when appropriate
- never below the configured visual floor without an explicit draft label

Blender evaluates the full continuous animation curve at each sparse timestamp. Motion blur is configured consistently to make the stylized cadence intentional. After compositing, the gap is normalized to the exact source frame count and frame rate.

The first implementation uses deterministic timestamp-based frame duplication/blending because it is stable and cheap. Motion-interpolation filters remain an optional visual experiment; they become default only after tests prove they do not warp silhouettes, paths, HUD text, or confidence overlays.

### Quality profiles

Profiles change render cost without changing evidence or story decisions:

| Profile | Purpose | Reconstruction FPS | Internal scale | Cycles samples | Entity budget |
|---|---|---:|---:|---:|---:|
| Storyboard draft | fast story/pose review | key poses or 6 fps | 35% | 2 | 6 |
| Standard forensic | default full reconstruction | 10–12 fps | 50% | 4 | 8 |
| Quality forensic | approved final/hero gaps | 12–15 fps | 65% | 8 | 10 |
| Workbench fallback | unsupported/failed GPU path | 10–12 fps | 50% | n/a | 8 |

The entity budget is the detailed-geometry budget, not permission to erase strongly supported entities. Supported overflow uses the documented simplified proxy tier. All profiles use the same camera, assets, materials, color management, and storyboard. Low-confidence entities are simpler by evidence policy, which also reduces render cost without hiding uncertainty.

### Adaptive gap policy

One global style is retained, but the renderer may reduce samples or entity detail for a complex gap when the reason is recorded. A deterministic complexity score uses:

- rendered entity count
- transparent/weak entities
- dynamic shadows
- moving camera
- environment proxy count
- projected overlap/occlusion
- output pixel count

The system never deletes a strongly supported entity merely to meet a time target. It first reduces sample count, internal scale, secondary geometry, and expensive effects. Any entity excluded by the evidence/readability budget remains listed in the report.

### Representative benchmark before full rendering

After the storyboard is accepted, the system selects:

- one highest-complexity gap
- one median-complexity gap when materially different

It first renders five diagnostic poses—start, 25%, midpoint, 75%, and end—then one complete representative gap at the intended Standard profile. The UI shows the story, contact sheet, full preview, seconds per sparse frame, predicted full runtime, and quality profile.

The full render remains blocked until:

- story and evidence references validate
- feet/wheels contact the ground
- identity, path direction, scale, camera, and occlusion read correctly
- the preview contains no blank/duplicate/corrupt layer
- timing projection fits the configured environment budget or the operator explicitly accepts a lower/higher profile

### Resumable frame contract

Blender writes individual PNG layers and an atomic `frame_manifest.json`; it does not make a gap valid merely because an MP4 exists.

Each frame record contains:

- gap and sparse-frame index
- source timestamp and intended source-frame range
- storyboard and render-profile digest
- scene-shell and asset digest
- expected layer names, dimensions, and hashes
- render duration and device
- validation status

On resume, only missing, invalid, or incompatible frames are rendered. Completed layers are recomposited without rerendering. A gap MP4 is encoded only after the complete manifest validates.

Colab renders from `/content`. It checkpoints completed gaps immediately and batches partial frame checkpoints into bounded archives before copying them to Drive, avoiding slow per-frame Drive writes. The manifest is copied atomically after its matching archive.

### CPU/GPU scheduling

- YOLO uses CUDA first and releases cached allocations before Blender.
- Cycles uses one T4 Blender worker; multiple Cycles processes on one T4 are not assumed to increase throughput.
- While the GPU renders the next sparse frame/gap, CPU workers may composite, validate, encode, and checkpoint the previous completed frames.
- Local Workbench/CPU rendering may use up to three measured workers when memory and benchmarks permit.
- RAM usage is bounded; decoded full videos and entire uncompressed frame sets are not retained in memory.

### Failure and fallback policy

- Azure gap planner unavailable/invalid: explicit deterministic hypothesis selection; story marked deterministic fallback.
- Azure narrative synthesizer unavailable: deterministic readable summary; accepted storyboard unchanged.
- Vision input unsupported: structured text evidence only; report the downgrade.
- Cycles GPU unavailable: benchmark Workbench fallback before continuing.
- One frame fails: retry that frame within the bounded retry policy, then stop the gap without discarding valid frames.
- Colab disconnects: resume from compatible local/Drive manifests.
- Full render estimate over budget: pause for profile selection; do not silently begin.
- Calibration or identity confidence below threshold: render uncertainty proxy or stop the affected gap according to policy.

### Target configuration contract

Policy values belong in `config/reconstruction_config.json`, not scattered constants. The target shape is:

```json
{
  "reasoning": {
    "enabled": true,
    "planner_schema_version": 2,
    "planner_batch_max_gaps": 4,
    "reasoning_effort": "medium",
    "request_timeout_seconds": 120,
    "max_output_tokens": 8000,
    "visual_evidence": {
      "enabled": true,
      "detail": "low",
      "max_global_keyframes": 8,
      "max_boundary_frames_per_side": 2,
      "max_entity_crops_per_track": 2
    },
    "narrative": {
      "enabled": true,
      "maximum_summary_words": 180,
      "maximum_story_points": 12
    }
  },
  "renderer": {
    "default_profile": "standard_forensic",
    "reconstruction_fps_target": 10,
    "production_scale_percent": 50,
    "cycles_samples": 4,
    "cycles_use_denoising": true,
    "max_gpu_render_workers": 1,
    "max_cpu_postprocess_workers": 2,
    "frame_checkpoint_batch_size": 24,
    "representative_pose_count": 5,
    "require_preview_gate": true,
    "require_budget_approval": true
  }
}
```

Exact maxima remain configurable and must be validated with named bounds. Changing reasoning schemas, image selection, story decisions, storyboard compilation, or render profiles invalidates only the appropriate downstream cache layer.

### Migration policy

- Existing v1 `evidence_ledger.json`, `decision_trace.json`, and gap-wide hypotheses remain readable for audit and fallback.
- New runs use explicitly versioned v2 clue, decision, narrative, storyboard, and frame-manifest contracts.
- A v1 decision cache can never be treated as a valid v2 per-entity decision.
- Resume rejects mixed schema versions instead of partially guessing compatibility.
- README and the UI must report the active evidence, story, storyboard, and renderer schema versions.

### Implementation sequence and approval gates

**Gate A — schemas and deterministic evidence**

Status: implemented and covered by automated hidden-frame and contract tests.

- implement clue catalog, visual-evidence manifest, entity-level hypotheses, gap decisions v2, narrative, and storyboard schemas
- prove hidden ranges cannot enter text or image evidence packs
- show deterministic clue bullets in the UI

**Gate B — Azure structured story**

Status: implemented with strict local fallback. Colab performs a small live structured-output deployment probe before expensive processing; the real user deployment must still pass this probe at run time.

- add visible image inputs and batched gap-planning requests
- validate per-entity selections and event beats
- add the presentation-only whole-video narrative pass
- show clue bullets, summary, gap timeline, alternatives, confidence, and unknowns

**Gate C — storyboard compiler**

Status: compiler and render-budget artifacts are implemented. Representative diagnostic layers and the complete heaviest-gap preview now provide the approval artifact.

- compile only validated IDs/tokens into renderer data
- prove prose changes cannot change animation
- produce a deterministic contact-sheet storyboard without a full animation render

**Gate D — smart renderer foundation**

Status: composite sparse PNG resume, exact timing normalization, bounded same-render diagnostic layers, representative timing, runtime refusal, and Colab visual approval are implemented. Reusable `.blend` shell reuse and full-frame independent layer manifests remain pending optimization.

- create reusable scene shell and layered render contracts
- render sparse PNG layers and atomic manifests
- normalize one synthetic gap to exact source timing
- resume after intentionally interrupting the middle of a gap

**Gate E — representative real-video gap**

- implementation automatically renders diagnostic poses and one complete representative gap
- operator reviews story-to-motion consistency, camera, contact, identity, shadows, occlusion, and timing in Colab
- measured seconds/frame and predicted full runtime are recorded before approval

**Gate F — three-gap validation**

- test simple, crowded/high-complexity, and low-confidence gaps
- verify adaptive fidelity remains visually consistent and epistemically honest

**Gate G — full one-video reconstruction**

- complete every gap from manifests
- preserve source duration, frame count, resolution, and audio
- expose the story and final output together in the UI

**Gate H — unseen judge-style video**

- process a different compatible video without per-video code changes
- record unsupported conditions instead of claiming flawless universal reconstruction

### Definition of done for this track

This track is complete only when:

- every displayed clue references visible evidence
- every material story statement references accepted clues/decisions
- every animated entity/action/path references a validated storyboard ID
- hidden frames remain unavailable until evaluation
- judges can read clue points and the whole reconstruction story in the UI
- one interrupted render resumes inside a gap without rerendering valid frames
- representative and three-gap visual gates pass
- the final video matches the source timing/audio contract
- Azure and deterministic fallback modes are visibly distinguishable
- the result is presented as evidence-grounded inference, never recovered truth

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
11. Before rendering, the UI exposes visible clue points, a whole-video reconstructed story, and an evidence-linked per-gap decision trace containing selected and rejected hypotheses, confidence, and unknowns.
12. Blender renders only the deterministic storyboard compiled from validated story decisions, never free-form model prose.

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
- an Azure OpenAI decision produced exclusively from the allowed visible evidence above

Forbidden inputs until evaluation:

- any hidden frame
- detections generated from hidden ground truth
- optical flow calculated through hidden frames
- appearance crops taken from hidden frames
- metrics from hidden truth used to change the current reconstruction
- the post-gap position used as a hard target to back-solve speed, force arrival, or reshape the predicted path
- model-generated track IDs, observations, coordinates, or events that cannot be traced to the validated evidence ledger or a deterministic candidate hypothesis

The post-gap observation is allowed only as a soft consistency check. The primary path must be predicted from pre-gap velocity, heading, acceleration limits, and scene constraints. Evaluation records how far that prediction is from the post-gap observation. The planner must not silently bend the path until that residual becomes zero.

Hidden truth may be read only after every gap has been rendered.

Azure OpenAI must be called before hidden-truth materialization. Its request artifact must contain an evidence-ledger digest and explicit visible-frame references. The response validator must reject unknown entity IDs, out-of-range timestamps, unsupported actions, unconstrained coordinates, missing evidence references, and confidence outside `0–1`. Azure output never bypasses the existing plan-v2 validator.

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
    +-- structured evidence ledger
    |       |
    |       +-- visible track/boundary facts
    |       +-- selected visible keyframes
    |       +-- deterministic candidate hypotheses
    |       +-- contradictions and unknowns
    |       |
    |       +-- optional Azure OpenAI reasoner
    |               |
    |               +-- strict structured decision trace
    |               +-- selected/rejected hypotheses
    |               +-- evidence references and confidence
    |
    +-- deterministic decision validator
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

Azure OpenAI is also not the renderer. It may summarize clues and rank bounded hypotheses, while deterministic Python remains authoritative for identities, coordinate conversion, physical limits, evidence isolation, schema validation, and final plan construction.

Blender's bundled Python must not import the system `src/` business-logic modules or depend on system site packages. System Python validates evidence and writes plain JSON contracts for plan v2, the global identity registry, calibration, and proxy geometry. Blender reads those contracts with its standard library and owns only `bpy` scene construction and rendering. A smoke test must verify this process and import boundary.

### 7.1 Azure OpenAI evidence-reasoning layer — implemented v1 baseline

**Status: first bounded implementation complete; live Azure and visual gates pending.** This v1 gap-wide selector is the safe baseline. Section 2.1 defines the required v2 upgrade to visible image evidence, per-entity hypotheses, renderable event beats, a whole-video narrative, and a deterministic storyboard compiler. Deterministic geometry remains authoritative and is also the explicitly labeled fallback.

The system will create one `evidence_ledger.json` per video before any Azure request. It will contain:

- global scene and camera observations derived from visible frames
- stable entity IDs, classes, lifecycle, and confidence
- per-gap pre/post boundary facts with source frame references
- measured velocity, heading, lifecycle, calibration, and continuity confidence
- deterministic candidate paths/actions with stable hypothesis IDs
- heading disagreement, post-gap residuals, and camera-calibration confidence
- a future, separately validated visible-keyframe extension; the current implementation sends structured evidence only

The Azure reasoner receives the ledger, not raw hidden footage. It returns one strict `decision_trace.json` with this conceptual shape:

```json
{
  "schema_version": 1,
  "evidence_digest": "sha256:...",
  "decisions": [
    {
      "gap_index": 0,
      "selected_hypothesis_id": "measured_continuation",
      "evidence_references": ["track:person_6:pre_boundary"],
      "decision_summary": "Continues forward at approximately the measured pre-gap pace.",
      "rejected_hypotheses": [
        {"id": "stationary_hold", "reason": "Conflicts with visible heading evidence."}
      ],
      "confidence": 0.71,
      "unknowns": ["Exact arm motion is not observable."]
    }
  ],
  "metadata": {"provider": "azure_openai", "deployment": "gpt-5.4"}
}
```

This trace is the public reasoning artifact shown in the UI. Raw private chain-of-thought must not be requested, persisted, logged, or presented. When the deployed reasoning model supports a reasoning summary, that supported summary may supplement—but never replace—the evidence references and schema fields.

The model's authority is deliberately narrow:

- it may summarize visible clues, rank supplied candidate hypotheses, identify contradictions, lower confidence, or abstain
- it may not invent entity IDs, read hidden frames, create arbitrary world coordinates, override physical limits, or silently force a path to meet post-gap evidence
- deterministic Python validates the evidence digest, references, IDs, value ranges, candidate membership, and plan-v2 contract
- an invalid response activates a clearly labeled deterministic fallback; a bounded repair request may be added later only if it remains measurable and safe

Use the Azure OpenAI Responses API with `store=False`. The implemented adapter uses Python's standard HTTPS library, avoiding another runtime dependency, and reads `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL`, and `AZURE_OPENAI_CHAT_DEPLOYMENT`. Azure API calls target the deployment name; the current local deployment value is `gpt-5.4`. Structured output is requested through a strict JSON schema and validated again locally.

Cache the response by evidence digest, deployment, prompt version, schema version, and inference configuration. A resume must never pay for or execute the same accepted reasoning request twice. API errors, rate limits, refusals, timeouts, token usage, and cache status belong in a sanitized reasoning report; secrets, full request authorization headers, and private image URLs do not.

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
    "mode": "generic_ground_prior",
    "motion_model": "dynamic_camera",
    "motion_applied_to_render": false,
    "calibration_confidence": 0.49,
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

**Current audited state:** this section remains the target design. The implementation measures visible-frame camera motion and robust height priors and can fit a per-video horizon/ground range from visible person-height and foot-contact observations when the fit has enough depth variation and low residual. It still does not solve focal length, a reviewed homography, or per-frame camera pose. Unsupported fits fall back explicitly to the generic prior. Dynamic-camera confidence is capped below `0.50`, and the render is labeled as a stabilized forensic view rather than source-camera matched.

### Camera model selection

The pipeline must classify each video as either `static_camera` or `dynamic_camera` before applying a ground-plane transform.

- `static_camera`: one reviewed calibration may be reused across the video.
- `dynamic_camera`: estimate visible-frame camera motion relative to a canonical evidence frame, then maintain a camera-pose track. This pose-track application is pending.

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

The target human system generates a clean low-poly human with an armature. The current implementation uses visible Blender primitives parented under articulated controls; it has walking limb motion but no skeletal armature yet. The target armature contains:

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
outputs/jobs/<job_id>/_work/<video>_<source_sha12>/entity_registry.json
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

**Status: pending.** The renderer currently provides a generic neutral grid and enables street proxy blocks only when a rendered vehicle supports that context. It does not yet author evidence-aligned geometry or validate proxy reprojection.

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
- one concise accepted gap-story line, prefixed as inference
- entity IDs when enabled
- optional legend for uncertainty colors

The HUD should be legible at 720p without covering important action. Debug paths and labels must be configurable separately from presentation mode.

## 16. Rendering Pipeline

Section 2.1 is the authoritative rendering design. This section records the Blender invocation and compatibility details that remain applicable beneath the storyboard-first, layered, sparse-frame renderer.

### Blender invocation

The normal Python process will invoke Blender in background mode:

```text
blender --background --python blender/render_gap.py -- \
  --plan <reconstruction_plan_v2.json> \
  --output <gap_video_or_preview> \
  --report <render_report.json> \
  --blend <scene.blend> \
  --mode <preview-or-animation>
```

### Render format

Recommended intermediate format:

- layered PNG frame sequences with transparency where required
- profile-controlled internal resolution with exact final upscale
- sparse reconstruction timestamps plus source-frame mapping metadata
- deterministic frame numbering

PNG sequences are restartable and avoid losing an entire render if a process stops. After rendering, frames will be encoded into a gap video. **Current state:** preview mode writes one PNG, while animation mode currently writes H.264 MP4 directly and resumes only at complete-gap granularity. Per-frame PNG resume remains a target improvement, not an implemented claim.

The direct-to-MP4 implementation is now a confirmed Colab risk: a T4 production run consumed roughly 3–4 hours and reached the session limit. The tiny 320×180 OptiX probe completed in 4.544 seconds but measured device availability, not representative scene throughput.

The revised production contract separates source FPS from reconstruction render FPS:

- preserve source FPS, duration, audio, and final frame count as immutable output requirements
- render Blender gaps at configurable `reconstruction_fps` in the 8–12 fps default range
- render the Standard profile at 50% internal resolution and 4 Cycles samples with denoising; use 8 samples only for the separately selected Quality profile
- deterministically resample the rendered gap to the exact source-frame count and upscale before stitching
- record render FPS, resampling method, scale, samples, measured seconds per rendered frame, and final contract in `render_report.json`
- use local PNG frame sequences and a validated frame manifest so resume can continue inside a gap instead of restarting it

Temporal resampling must never change entity paths, gap duration, or evidence timestamps. It is a rendering optimization, not an inference operation.

### Render engine

Local rendering starts with Blender Eevee when the graphics driver supports it, with Workbench as the compatibility fallback. Colab may use Cycles OptiX/CUDA only after a real device probe. The current 75%-scale, 16-sample, source-FPS Cycles configuration is rejected as the full-video default after exceeding the observed Colab runtime limit. One representative gap must establish a measured estimate before the full render; a projected run beyond the configured budget requires an explicit operator override.

### Cache policy

Each gap receives a cache signature based on:

- reconstruction plan hash
- Blender script version
- Blender configuration
- resolution and frame rate
- reconstruction render FPS, resampling policy, Cycles samples, and internal resolution scale
- evidence-ledger digest and accepted decision-trace digest

Unchanged gaps should not be rerendered.

**Current audited cache contract:** reuse verifies the exact plan SHA-256, animation mode, engine, frame count, production resolution, frame rate, and non-empty artifacts. Automatic Blender-script version hashing, evidence/decision digests, and per-frame resume remain pending; code changes that do not alter the plan should therefore use a fresh run identifier or explicitly bump the renderer contract.

## 17. Encoding and Audio

The existing OpenCV stitcher drops audio and produces inefficient `mp4v` output.

The implemented pipeline uses FFmpeg to:

- encode H.264 video
- preserve the original frame rate
- concatenate visible and reconstructed segments without changing duration
- encode source audio when present while constraining output duration to the exact reconstructed video duration
- produce a broadly compatible MP4

FFmpeg is not currently discoverable on this workstation. Installation remains an operator action; runtime preflight now reports the dependency before detection or Blender rendering begins.

## 18. Implemented and Target Files

The following map contains both implemented core files and explicitly pending calibration/proxy authoring modules. A listed target is not evidence that it exists.

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
    evidence_reasoning.py
    narrative_pipeline.py           # target: planner and summary orchestration
    storyboard_compiler.py          # target: decisions to renderer contract
  domain/
    configuration.py
    evidence_reasoning.py
    clue_catalog.py                 # target
    evidence_images.py              # target
    gap_hypotheses.py               # target: per-entity candidates
    reconstruction_narrative.py     # target
    render_storyboard.py            # target
    render_frame_manifest.py        # target
    processing_job.py
    video_upload.py
  interfaces/
    http/
      local_server.py
  infrastructure/
    azure_openai_reasoner.py
    environment.py
    render_checkpoint_store.py      # target
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
  build_scene_shell.py              # target
  render_sparse_layers.py           # target
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

Existing OpenCV compositing remains available through the UI's explicit `Fast 2.5D fallback` selection. The local CLI controls only host, port, and browser launch; it does not expose a `--renderer` flag.

## 19. Proposed CLI

```bash
python app.py
python app.py --host 127.0.0.1 --port 8000
```

The browser UI is the supported operator surface. The reconstruction orchestrator lives in `src/application/reconstruction_pipeline.py`; the obsolete root `run.py` entrypoint is removed. Renderer selection is already exposed in the UI, while a dedicated preview-gap control remains pending.

## 20. Implementation Phases and Approval Gates

### Phase 0 — Freeze and benchmark the baseline (local baseline implemented; Colab production benchmark reopened)

Work:

- preserve current 2.5D renderer as fallback
- record current output metrics and sample frames
- define a fixed seed and fixed first preview gap
- add renderer selection to configuration, without changing default behavior
- verify that Blender background mode can consume a system-Python-generated JSON fixture without importing `src/`
- benchmark one representative frame and one short sequence in Eevee and Workbench; record seconds per frame, memory use, and visible quality defects before choosing the default engine
- treat the 4.544-second 320×180 T4 OptiX render as a device probe only and replace it with representative-gap timing before another full Colab run

Gate:

- the full automated suite passes
- current output remains reproducible
- Blender starts in background mode, respects the JSON boundary, and completes a low-resolution render
- the selected render engine and benchmark evidence are recorded in the render configuration

### Phase 1 — Plan v2 and coordinate conversion (substantially implemented; true calibration pending)

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

### Phase 1A — Evidence ledger and Azure reasoning v1 (implemented; live validation pending)

Work:

- convert deterministic visible-only outputs into one strict evidence ledger
- generate bounded candidate hypotheses before the model call
- add the Azure Responses API adapter and require the configured deployment name
- request strict structured decisions, never raw chain-of-thought
- validate every evidence reference, entity ID, hypothesis ID, confidence, and unknown before plan-v2 construction
- cache accepted decisions and sanitize all error/usage reporting
- display extracted clues and the structured decision trace in the local UI

Gate:

- an Azure request contains no hidden frame, hidden detection, hidden metric, or secret in logs
- malformed, invented, out-of-range, or unreferenced model output is rejected
- the same evidence/configuration reuses the cached decision without another paid request
- API timeout, refusal, rate limit, and invalid schema produce a clean, visible state
- deterministic fallback is labeled explicitly and never masquerades as Azure-assisted reasoning
- judges can inspect the selected hypothesis, evidence references, rejected alternatives, confidence, and unknowns before rendering

### Phase 1B — Evidence narrative and story-to-storyboard v2 (next implementation gate)

Work:

- create the deterministic clue catalog and bounded visible image-evidence manifest
- replace gap-wide motion selection with per-entity hypotheses and lifecycle-compatible action tokens
- add strict gap decisions containing chronological renderable event beats
- validate every clue, entity, action, path, timestamp, and evidence reference locally
- compile accepted decisions into a renderer-only storyboard
- add the presentation-only whole-video narrative synthesizer
- show clue points, whole-video summary, and Before/Inside/After gap timeline in the UI

Gate:

- hidden frames cannot enter text or image inputs
- every story statement is traceable to clues or accepted decisions
- every animation instruction is traceable to a validated storyboard ID
- prose edits cannot change the storyboard
- Azure planner and narrative failures use separately labeled deterministic fallbacks
- the story and storyboard remain consistent across every gap for repeated identities

### Phase 2 — Single-frame Blender scene (partial; visual gate must be repeated)

Work:

- build camera, ground plane, lighting, HUD, and proxy environment
- author proxy geometry through the calibrated evidence camera with the visible backplate overlaid (pending)
- save the proxy contract and validate its wireframe projection against visible evidence (pending)
- build procedural person and vehicle meshes
- render a single midpoint frame for one gap

Gate:

- user reviews one rendered frame
- perspective, scale, color, labeling, and style are approved
- calibration confidence and its warning state are visible in debug HUD mode
- proxy wireframes align with evidence geometry within the configured screen-space tolerance

No animation work should proceed before this visual gate.

### Phase 3 — One-gap animation (partial; final user approval gate pending)

Work:

- add human armatures and walking cycles (primitive articulation exists; armatures remain pending)
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

- run the representative-gap benchmark and calculate the projected full-render duration
- require approval or an explicit runtime-budget override before scheduling all gaps
- render all gaps for `input_vid3.mp4`
- use sparse reconstruction FPS, reduced internal scale/samples, and resumable PNG frames under the Colab profile
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
- every Azure-selected hypothesis references only validated ledger evidence and a deterministic candidate ID
- every UI clue has at least one valid visible evidence reference
- every material whole-video story point references clue IDs or accepted gap decisions
- every renderable gap entity has its own selected hypothesis or explicit unresolved proxy
- every storyboard beat uses only validated entity, action, path, and asset IDs
- presentation-summary text cannot modify paths, actions, camera, or animation
- rejected alternatives, confidence, and unknowns remain visible in the structured decision trace
- raw model chain-of-thought is never requested, logged, persisted, or presented

### Technical

- exact original frame count
- exact original frame rate within container precision
- original resolution
- source audio preserved when present
- deterministic output for a fixed seed and configuration
- resumable per-frame layered renders with atomic manifests
- Blender failure returns a clean error and preserves logs
- no new unapproved external asset dependencies
- Azure reasoning is content-addressed and does not repeat an accepted paid request on resume
- sparse gap renders resample to the exact source-frame count without changing gap duration
- static environment passes are reused only under a matching camera/environment digest
- one T4 Cycles worker can overlap with bounded CPU composite/checkpoint work without corrupting manifests
- representative-gap timing produces a full-run estimate before all gaps are scheduled

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

Risk: even GPU Cycles can exceed the Colab T4 session limit when approximately 25% of a two-minute source is rendered at source FPS, 75% resolution, and 16 samples. The successful 4.544-second 320×180 probe does not predict full-scene throughput.

Mitigation:

- benchmark one representative production gap and project total time before the full run
- use 10–12 reconstruction fps, 50% internal scale, 4 Cycles samples, and denoising for the Standard Colab profile
- resample and upscale deterministically to the exact source contract after rendering
- use resumable PNG frames and validated frame manifests rather than direct-to-MP4 gap rendering
- retain Eevee or Workbench where their graphics paths are actually faster and supported
- reuse the content-addressed scene shell and static environment pass when the camera contract permits
- overlap one T4 render worker with CPU compositing, validation, encoding, and checkpointing
- refuse an over-budget full render unless the operator explicitly overrides the estimate

### Model-generated unsupported claims

Risk: an Azure model produces a plausible narrative that is not supported by the visible 75%.

Mitigation:

- send a compact evidence ledger plus bounded candidate hypotheses instead of an open-ended request to imagine the gap
- require stable evidence references and candidate IDs in strict structured output
- reject unknown entities, arbitrary coordinates, unsupported actions, and missing references
- keep deterministic geometry and physical constraints authoritative
- expose selected and rejected hypotheses, confidence, and unknowns to the operator before rendering
- abstain or use a visibly labeled deterministic fallback when the model cannot support a decision

### Azure availability, privacy, and cost

Risk: the configured deployment is missing, unavailable, rate-limited, expensive, or unsuitable for sending visible video evidence.

Mitigation:

- require `AZURE_OPENAI_CHAT_DEPLOYMENT` with `AZURE_OPENAI_BASE_URL` and the API key before Azure-assisted processing
- minimize and resize visible keyframes; never send hidden frames or unnecessary full video
- disclose external Azure processing in the UI
- use `store=False`, sanitized logs, bounded timeouts/retries, token limits, and content-addressed caching
- never expose the API key or authorization headers

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
- evidence-ledger schema, stable digest, and hidden-frame rejection
- Azure structured-decision parsing, refusal, timeout, rate-limit, and malformed-output fallback behavior
- invented entity/evidence/hypothesis rejection and confidence-range validation
- Azure decision-cache reuse without a second API call
- visible image-manifest validation and hidden-frame rejection
- clue-catalog determinism, traceability, and conflict reporting
- per-entity hypothesis and lifecycle/action compatibility
- ordered gap event-beat validation
- whole-video narrative sentence-to-clue/decision traceability
- causal-link rejection when only chronological, independent events are supported
- proof that presentation prose cannot alter the renderer storyboard
- layered sparse-frame manifest integrity and interrupted-frame resume
- reconstruction-FPS resampling to exact source frame counts

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
outputs/jobs/<job_id>/_work/<video>_<source_sha12>/gaps/gap_XX/blender/
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
17. The UI shows the evidence ledger and structured decision trace before Blender rendering begins.
18. Azure-assisted output is reproducible from cached decisions, and Azure failures or deterministic fallback are labeled explicitly.
19. A representative-gap benchmark predicts a run within the configured budget before the full video is scheduled.
20. Interrupted Colab rendering can resume from validated completed frames, not only completed gaps.
21. The UI shows deterministic clue points and a whole-video reconstructed story with inspectable references.
22. Every renderable entity decision is per-entity rather than one motion choice applied to an entire gap.
23. Every event beat and animation parameter originates from a validated storyboard ID, never free-form prose.
24. The scene shell and static environment layers are reused when—and only when—the camera/environment contract permits reuse.
25. Blender renders the Standard profile at sparse reconstruction timestamps and normalization restores the exact source timing contract.
26. An intentionally interrupted representative gap resumes without rerendering validated frames.

## 26. Approved Implementation Decisions

The following decisions are approved for implementation and remain change-controlled:

1. **Visual style:** use deliberate stylized forensic 3D as the default.
2. **Renderer:** Blender 4.5 LTS through checked-in `bpy` scripts, not Blender MCP.
3. **Assets:** procedural humans and vehicles first; no downloaded asset packs.
4. **Calibration:** allow one reviewed per-video calibration file for judge-facing inputs.
5. **Review sequence:** one frame, then one gap, then three gaps, then one full video.
6. **FFmpeg:** FFmpeg and FFprobe are required runtime tools; installation is an operator action and local reconstruction must stop at preflight when either is unavailable.
7. **Fallback:** keep the current 2.5D renderer available but never silently substitute it in a Blender judge render.
8. **Claims:** call the output an AI-inferred forensic reconstruction, not recovered footage.
9. **Prediction:** use pre-gap motion as the primary path estimate and post-gap observations only as soft consistency checks.
10. **Identity:** generate and cache appearance and body parameters once per global track ID, then reuse them across every gap.
11. **AI authority:** Azure OpenAI may summarize evidence and rank bounded hypotheses; deterministic Python remains authoritative for evidence isolation, IDs, coordinates, physics, validation, and plan construction.
12. **Reasoning visibility:** expose a structured decision trace with evidence references, selected/rejected hypotheses, confidence, and unknowns; never request or expose raw private chain-of-thought.
13. **Azure configuration:** read key, base URL, and deployment from environment variables; never hardcode or persist secrets.
14. **AI failure:** reject an invalid response and activate the explicitly labeled deterministic fallback; consider one bounded repair attempt only after live telemetry justifies it.
15. **Colab performance:** require a representative-gap estimate, sparse rendering, and per-frame resume before another full two-minute Cycles run.
16. **Story contract:** “story” means evidence-grounded reconstruction narrative, never unrestricted fiction; every material statement is traceable.
17. **Executable boundary:** only validated gap decisions and the deterministic storyboard influence Blender; the readable whole-video summary is presentation-only.
18. **Rendering architecture:** use one reusable scene shell, layered sparse PNG passes, exact final timing normalization, and frame-level resume.

## 27. Local Processing UI

### Goal

Provide one clean local interface that a judge or operator can use without terminal commands. Processing time is allowed, but the interface must continuously communicate what the system is doing and must never appear frozen.

### Input workflow

- support browsing and drag-and-drop upload
- accept common video containers: MP4, MOV, AVI, MKV, M4V, WebM, MPEG/MPG, and WMV
- sanitize filenames, reject unsupported extensions, enforce upload size/transfer bounds, validate finite media metadata, and reject sources over 10 minutes, 120 fps, a 4096-pixel side, or a 3840×2160 total-pixel budget before queueing expensive work
- require even dimensions and constant frame rate for the frame-indexed H.264 pipeline; fail cleanly rather than silently changing timing or geometry
- store each upload and its output in a unique job directory so repeated filenames cannot overwrite one another
- process one reconstruction job at a time by default because detection and rendering are CPU/GPU intensive

### Processing experience

- show queued, validating, selecting gaps, detecting/tracking, extracting clues, building hypotheses, planning gaps, validating decisions, summarizing story, compiling storyboard, benchmarking, previewing, rendering layers, compositing, normalizing, evaluating, stitching, cancelling, cancelled, completed, and failed stages
- expose an evidence-clues view after deterministic extraction, grouped into scene, entity, per-gap boundary, conflict, and unknown sections
- expose a whole-video **Reconstructed story** summary with global confidence, cited story points, and explicit unknowns
- expose a Before-observed / Inside-inferred / After-observed timeline for every gap
- expose the structured decision trace with per-entity selected hypotheses, renderable event beats, evidence references, rejected alternatives, confidence, and unknowns before Blender rendering begins
- show whether the job is Azure-assisted, deterministic, or using an explicitly acknowledged fallback
- disclose that selected visible evidence may be sent to the configured Azure resource when Azure-assisted mode is enabled
- expose real stage progress from the Python pipeline rather than a purely decorative percentage
- show elapsed time and a clearly labeled best-effort ETA derived from observed progress; count it down between pipeline updates and switch to `recalculating` when an in-flight task exceeds the latest estimate
- provide an expandable live-activity panel per job with completed, active, and pending stages, per-stage percentage, timestamps, and recent persisted pipeline messages
- stream Blender frame markers from every active worker into aggregate render progress so a long individual gap does not leave the percentage visually frozen
- treat the Blender safety timeout as an inactivity threshold that resets on every frame marker, never as a fixed total render-duration limit
- open the live-activity panel automatically for a newly submitted reconstruction
- poll local job state without reloading the page and preserve completed job metadata across server restarts
- surface clean errors in the UI while retaining detailed local logs
- expose a cancel action for queued and processing jobs; show `cancelling` until active Blender/FFmpeg processes have stopped
- keep one top-level video job active at a time; allow up to three measured local CPU/Workbench workers, but use one Cycles worker on a single T4 while CPU workers composite and checkpoint completed frames
- show representative-gap seconds per frame, projected full-render duration, active runtime budget, and the operator's approval/override state

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

Video responses must support bounded HTTP range requests so browser seeking works. The UI/API bind only to loopback, validate the Host header, permit one active upload, close stalled request sockets during shutdown, and use a project lock to prevent two local servers from mutating the same jobs. They add no network service or frontend framework dependency.

### UI acceptance gate

- a supported test video can be uploaded without using the terminal
- progress, current stage, elapsed time, and ETA update while processing
- the ETA continues changing between stage updates and clearly reports when it is recalculating
- the live-activity dropdown identifies completed, active, and pending work and preserves its open state across polling refreshes
- extracted clue points, the whole-video reconstructed story, and the structured per-gap decision trace can be inspected before the render is committed and remain available after completion
- observed, inferred, unknown, and rejected information are visually distinct
- every displayed material story point has inspectable evidence/clue references
- invalid Azure output and deterministic fallback are clearly visible rather than hidden behind generic progress text
- an over-budget projected render pauses for explicit approval instead of consuming the full runtime automatically
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

- refuse to start expensive work when Colab has not assigned an NVIDIA GPU or PyTorch cannot use CUDA
- report Blender's graphics vendor/renderer separately and verify Cycles GPU access with a real OptiX/CUDA render instead of inferring it from Xvfb
- install the pinned Blender 4.5 LTS binary, project Python requirements, and FFmpeg inside the ephemeral runtime
- process from `/content` instead of directly against mounted Drive
- accept one validated common-format video and derive a run identifier from video content, seed, effective configuration, Git commit, and Blender version
- show live stage, detail, and aggregate progress emitted by the shared pipeline
- import Azure credentials into the Colab runtime only through a secret/environment mechanism; never write them into the notebook or Drive artifacts
- run the same evidence-ledger, Azure decision, validation, and caching code as the local pipeline when Azure-assisted mode is enabled
- run YOLO and Blender sequentially on the same T4, explicitly releasing YOLO's cached CUDA allocations before the Blender phase
- treat the small OptiX/CUDA render only as a device probe, then benchmark one representative reconstruction gap before the full run
- use the Standard profile: 10–12 reconstruction fps, 50% scale, 4 samples, denoising, one T4 worker, and at most eight detailed entities
- calculate and show a full-run estimate; require an explicit override when it exceeds the configured Colab budget
- fall back automatically to a two-worker Colab-safe Workbench profile when both GPU probes fail, and terminate a Blender gap after 15 minutes without a completed-frame marker
- checkpoint validated PNG frame manifests as well as complete gap artifacts so interruption can resume inside a gap
- reuse the compatible scene shell and render layers, and overlap GPU rendering with CPU compositing/checkpoint work
- restore compatible completed-frame/gap checkpoints only when evidence, decision, plan, render, and source-content contracts match
- save the final video and JSON reports to Google Drive before offering download; skip inline embedding for results over 80 MB
- clearly state that Colab resources and runtime duration are not guaranteed

The notebook does not start `app.py`, expose a tunnel, or replace the local judge-facing browser UI. Its purpose is durable cloud batch execution with CUDA-accelerated YOLO and explicitly reported Blender device behavior while preserving the same evidence and rendering contracts.
