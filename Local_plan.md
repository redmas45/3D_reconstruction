# AI Evidence Gap Reconstruction - Local Execution Plan

## Correct Product Goal

Input: every video inside `data/input`.

For each video:

1. Split the video into 4 equal chunks.
2. Randomly choose 1 chunk as the missing evidence chunk on every run.
3. Keep the other 3 chunks as original evidence footage with live YOLO classification overlays.
4. Analyze the visible 75% deeply.
5. Reconstruct only the missing 25% as an animated 3D-style inference scene.
6. Output one full-length reconstructed video per input video.

The goal is not a skeleton popping into the scene and not pasted ghost footage. The goal is intelligence-driven reconstruction: detect what exists, show those detections live with YOLO boxes on the visible chunks, understand movement and object continuity, infer what likely happened in the missing chunk, then render the missing part as an animated 3D reconstruction.

---

## What The System Must Deduce

From the 3 visible chunks, the system should infer:

- how many people are present
- where each person appears before and after the missing chunk
- movement direction and approximate path
- whether a person is carrying a bag, backpack, handbag, suitcase, or other visible item
- vehicles or large objects present in the scene
- object-person associations, such as "person 2 carries backpack"
- which entities continue across the gap
- which entities enter or leave the frame
- the most likely state of the scene during the hidden chunk

This intelligence should be written to a machine-readable report before rendering:

```text
outputs/_work/<video_name>/scene_report.json
outputs/_work/<video_name>/reconstruction_plan.json
```

---

## Reconstruction Rule

The visible 75% must remain evidence footage, with YOLOv8m bounding boxes and class labels overlaid.

Only the randomly hidden 25% becomes a generated reconstruction view.

Final output:

```text
outputs/<input_video_name>_reconstructed.mp4
```

For the current two lower-resolution videos:

```text
outputs/input_vid3_reconstructed.mp4
outputs/input_vid4_reconstructed.mp4
```

---

## Revised Local Project Structure

```text
3D reconstuction/
|
|-- run.py                         # single entrypoint
|-- README.md
|-- Local_plan.md
|-- requirements.txt
|
|-- data/
|   `-- input/                     # all source videos
|
|-- src/
|   |-- detect.py                  # YOLOv8m detection
|   |-- track.py                   # multi-object tracking and identity continuity
|   |-- scene_intelligence.py      # people/object summaries and direction inference
|   |-- gap_selector.py            # 4 chunks, choose 1 random missing chunk
|   |-- reconstruction_plan.py     # converts detections/tracks into missing-chunk plan
|   |-- visual_output.py           # YOLO overlays + animated 3D reconstruction renderer
|   |-- stitch.py                  # visible chunks + reconstructed chunk
|   `-- evaluate.py                # compares against hidden ground truth
|
`-- outputs/
    |-- input_vid3_reconstructed.mp4
    |-- input_vid4_reconstructed.mp4
    `-- _work/
        `-- <video_name>/
            |-- chunks/
            |-- detections.json
            |-- tracks.json
            |-- scene_report.json
            |-- reconstruction_plan.json
            |-- chunk_hidden_3d_reconstruction.mp4
            `-- accuracy_report.json
```

No notebooks are needed. The workflow should be one command:

```bash
python run.py
```

---

## Detection And Tracking Plan

Use the strongest practical Ultralytics model available locally. The active model is configured as:

```json
"model": "yolo26m.pt"
```

This is configured in:

```text
config/reconstruction_config.json
```

Current inference parameters:

```json
{
  "model": "yolo26m.pt",
  "frame_stride": 8,
  "downscale_width": 960,
  "confidence": 0.3
}
```

Lower `frame_stride` gives smoother live boxes and better tracks, but costs more time. Higher `downscale_width` can improve small-object detection, but costs more time.

Confidence must be between `0` and `1`. If a value greater than `1` is entered, the runner treats it as a percentage. Example: `3` becomes `0.03`.

All 80 COCO classes are enabled:

- person
- car
- motorcycle
- bicycle
- bus
- truck
- backpack
- handbag
- suitcase
- bottle
- cup
- knife
- cell phone
- sports ball or other relevant carried object when detected

Detection must run on all 3 visible chunks, not only the immediate boundary frames.

Tracking requirements:

- assign stable IDs to people and relevant objects
- estimate each tracked entity's path
- estimate direction: left, right, toward camera, away from camera, stationary
- preserve person-object associations when close enough over time
- summarize start and end states around the missing chunk

Expected report example:

```json
{
  "people_count": 2,
  "objects": ["backpack", "car"],
  "tracks": [
    {
      "id": "person_1",
      "class": "person",
      "direction": "right",
      "visible_before_gap": true,
      "visible_after_gap": true,
      "associated_objects": ["backpack_1"]
    }
  ]
}
```

---

## Gap Selection Rule

For each video:

1. Read total frame count.
2. Split frame range into 4 equal chunks.
3. Randomly choose one chunk as hidden.
4. Save the hidden chunk as ground truth for later comparison only.
5. Never use hidden frames during reconstruction.

This means the missing segment is always approximately 25% of the video.

Use `--seed` only when repeatability is needed:

```bash
python run.py --seed 123
```

Without `--seed`, each run should hide a different chunk.

---

## Reconstruction Strategy

The renderer should use a reconstruction plan, not raw skeleton interpolation.

Minimum acceptable behavior:

1. Render visible chunks with live YOLOv8m boxes, class names, IDs, and direction hints.
2. For each tracked person/object, estimate where it should be during the missing chunk.
3. Convert inferred paths into a 3D-style ground-plane animation.
4. Draw people, vehicles, and motion trails as labeled reconstructed entities.
5. Include a HUD saying the chunk is AI 3D reconstruction / missing evidence.
6. Keep object/person continuity consistent across the gap boundaries.
7. Avoid visible skeleton overlays in the final evidence chunks.

Better future rendering options:

- optical-flow-guided interpolation near gap boundaries
- object sprite warping using segmentation masks
- inpainting for removed/occluded areas
- local image-to-video model if the machine can support it
- external generative video model only if explicitly allowed later

Important: if the system is uncertain, the output should be conservative and plausible, not visually loud or random.

---

## Evaluation

Evaluation uses the hidden 25% only after reconstruction is complete.

Metrics:

- SSIM / PSNR for frame similarity
- object center trajectory error
- detection consistency between reconstructed and real hidden chunk
- person count consistency
- object count consistency
- boundary smoothness at chunk joins

The old joint-only error is not enough because the target is full-scene reconstruction, not only human pose.

---

## Implementation Phases

### Phase 1 - Fix Product Contract

- Make `run.py` the only supported entrypoint.
- Remove notebook flow.
- Split videos into 4 chunks.
- Randomly hide 1 chunk.
- Keep the other 3 chunks exact.
- Output one final video per input video.

### Phase 2 - Scene Intelligence

- Switch from YOLOv8n to YOLOv8m.
- Detect people, vehicles, bags, and relevant objects.
- Run detection over all visible chunks.
- Track entity IDs across chunks.
- Build `scene_report.json`.

### Phase 3 - Reconstruction Planning

- Infer entity paths across the missing chunk.
- Associate bags/objects with people.
- Decide who/what should appear in the missing chunk.
- Write `reconstruction_plan.json`.

### Phase 4 - Realistic Gap Rendering

- Build a background plate from visible chunks.
- Use real entity crops from visible frames.
- Move/scale/blend entities along planned paths.
- Add shadows, feathering, and boundary smoothing.
- Avoid visible skeleton overlays in final output.

### Phase 5 - Evaluation And Iteration

- Compare generated chunk with hidden ground truth.
- Measure object count, trajectory, SSIM, and boundary smoothness.
- Run multiple passes and keep the best-scoring reconstruction.

### Phase 6 - Statement-Aware Reconstruction

Future statement feature:

- Parse text descriptions into structured clues.
- Examples: red shirt, phone in hand, cup in hand, knife in other hand, ran left, entered car.
- Match statement clues against YOLO detections and visible appearance.
- Use the matched clues to alter the reconstruction plan.
- Do not blindly trust statements when the video contradicts them; preserve both video evidence and witness claim separately.

Example:

```json
{
  "person_attributes": {"shirt_color": "red"},
  "held_objects": ["phone", "knife"],
  "motion": "walking right",
  "confidence": 0.72
}
```

---

## What Went Wrong In The Previous Prototype

The earlier prototype focused too much on pose/skeleton interpolation. That made the output look artificial and confusing, especially when pose detections were weak or stale near the gap boundary.

The corrected project direction is broader:

- first understand the scene
- then infer missing evidence
- then render the hidden chunk

Skeletons may remain useful internally, but they should not be the main visible output.
