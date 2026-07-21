# AI-Inferred Evidence Visualization

Local pipeline for replacing a distributed 25% of each input video with evidence-grounded, stylized Blender forensic-3D inference views. The former 2.5D compositor remains available as an explicit compatibility fallback.

## Product Contract

For every video in `data/input`, the pipeline:

1. Randomly selects multiple non-overlapping gaps.
2. Keeps every gap between 1 and 3 seconds.
3. Makes the combined hidden duration approximately 25% of the full video.
4. Keeps the remaining 75% as visible evidence with live YOLO classifications.
5. Tracks people, vehicles, bags, and relevant objects through the visible ranges.
6. Reconstructs each short gap from evidence immediately before and after it.
7. Stitches the original timeline back together at its original duration.
8. Evaluates the completed reconstructions against hidden ground truth afterward.

The generated portions are explicitly labeled **AI-inferred evidence visualization — not ground truth**.

## Reconstruction Style

The default hidden-gap renderer uses headless Blender 4.5 LTS:

- perspective-matched forensic ground and city-street proxy geometry
- procedural articulated people and simplified vehicles
- global per-track body proportions and evidence-derived clothing colors
- forward-predicted three-waypoint paths; post-gap positions remain soft residuals
- confidence-based solid, translucent, or simplified silhouette fidelity
- evidence inset, calibration confidence, uncertainty rings, and explicit non-ground-truth labeling
- short dark forensic shutters between visible evidence and inferred 3D

The UI defaults to `Blender Forensic 3D`; `Fast 2.5D fallback` must be selected explicitly. Blender failures stop a Blender job and never silently substitute 2.5D output.

## Output

```text
outputs/jobs/<job_id>/<video_name>_reconstructed.mp4
outputs/jobs/<job_id>/job.json
outputs/jobs/<job_id>/_work/<video_name>/gap_selection.json
outputs/jobs/<job_id>/_work/<video_name>/detections.json
outputs/jobs/<job_id>/_work/<video_name>/scene_report.json
outputs/jobs/<job_id>/_work/<video_name>/entity_registry.json
outputs/jobs/<job_id>/_work/<video_name>/camera_motion_report.json
outputs/jobs/<job_id>/_work/<video_name>/reconstruction_plans_v2.json
outputs/jobs/<job_id>/_work/<video_name>/gaps/gap_XX/blender/plan_v2.json
outputs/jobs/<job_id>/_work/<video_name>/gaps/gap_XX/blender/scene.blend
outputs/jobs/<job_id>/_work/<video_name>/gaps/gap_XX/blender/gap_blender.mp4
outputs/jobs/<job_id>/_work/<video_name>/gaps/gap_XX/blender/render_report.json
outputs/jobs/<job_id>/_work/<video_name>/accuracy_report.json
```

Hidden ground-truth segment files are not created during preparation. They are materialized only after every inferred gap has rendered, then used for evaluation. They are never passed to detection, tracking, planning, appearance sampling, camera estimation, or Blender.

## Local Interface

Prerequisites on this workstation are Python 3.12, Blender 4.5 LTS, and FFmpeg 8.1. The Python dependencies remain listed in `requirements.txt`.

```bash
pip install -r requirements.txt
python app.py
```

The interface opens at `http://127.0.0.1:8000` and provides:

- browse and drag-and-drop upload for common video formats
- a persisted one-at-a-time processing queue
- expandable live activity with completed, active, and pending stages plus persisted pipeline messages
- per-frame Blender worker updates aggregated across all active gap renders
- live progress and elapsed time with an ETA that counts down between updates and reports when it is recalculating
- responsive cancellation for queued jobs, Python stages, Blender gap workers, and FFmpeg encoding
- in-browser playback and download for new jobs and existing reconstructed outputs
- confirmed deletion of the output, work directory, and retained upload
- persistent dark/light theme toggle in the top-right navigation
- selectable Blender forensic-3D or explicit 2.5D fallback rendering

Useful server options:

```bash
python app.py --host 127.0.0.1 --port 8000
python app.py --no-browser
```

The UI uses only Python's standard-library HTTP server and vanilla HTML, CSS, and JavaScript. No web framework or extra runtime dependency is required.

## Google Colab

Open or upload `colab/reconstruction.ipynb` in Google Colab, select a GPU runtime, and run its cells in order. The notebook clones this repository and calls the same `src/`, `blender/`, and `config/` pipeline used by the local interface; it does not maintain a second reconstruction implementation or expose the local web UI through a tunnel.

The notebook installs Blender 4.5 LTS and FFmpeg, verifies that Colab assigned an NVIDIA GPU, accepts one common-format video upload, renders from fast `/content` storage, checkpoints completed Blender gaps to Google Drive, saves final reports and video under `MyDrive/3D_Reconstruction`, and offers a final download. Push local changes to the `main` branch before starting so the cloned runtime uses the current pipeline. If the assigned runtime runs out of memory, reduce the notebook's parallel Blender worker setting from two to one.

## Project Structure

```text
app.py                              Local UI entrypoint
src/application/                    Pipeline and processing-job orchestration
src/domain/                         Validated configuration, job, and upload policies
src/interfaces/http/                Local HTTP API and static-file boundary
src/infrastructure/                 Blender, visible-frame, camera-motion, and FFmpeg adapters
blender/                            JSON-only procedural scene, animation, HUD, and render scripts
src/*.py                            Existing detection, inference, rendering, and evaluation capabilities
web/index.html                      Accessible application shell
web/assets/styles/                  Professional dark/cyan visual system
web/assets/scripts/                 Typed API client, formatters, and UI controller
colab/reconstruction.ipynb          Single-file Colab GPU batch interface
tests/unit/                         Layered domain, application, and interface tests
tests/test_*.py                     Existing reconstruction behavior tests
```

New modules follow the engineering standards in `rules.md`: explicit types, validated boundaries, named policy constants, focused functions, and files below the 500-line source budget.

## Configuration

The primary settings are in `config/reconstruction_config.json`:

```json
{
  "gap": {
    "missing_fraction": 0.25,
    "min_seconds": 1.0,
    "max_seconds": 3.0,
    "context_seconds": 2.0
  },
  "renderer": {
    "default_mode": "blender",
    "blender_version": "4.5 LTS",
    "production_scale_percent": 100,
    "max_parallel_gap_renders": 3
  }
}
```

The UI accepts fractional source rates such as 29.97 and 59.94 fps. The job manager keeps one full video reconstruction active at a time. Within that job, a bounded three-thread pool launches up to three independent Blender subprocesses for separate gaps. Raise this setting only after measuring memory and render-device pressure. Source frames remain streamed instead of being held as an uncompressed full-video RAM cache.

YOLO uses sequential BoT-SORT tracking with camera-motion compensation. The scene-intelligence stage performs dependency-free appearance matching across gaps. Tracker configuration lives in `config/botsort_reid.yaml`. Confidence values must be between `0` and `1`; values greater than `1` are interpreted as percentages.

No OpenAI API key is used by the current deterministic reconstruction pipeline.

## Evaluation

Only after every missing gap has been rendered, the evaluator reads hidden ground truth and reports:

- SSIM and PSNR
- entry and exit boundary similarity
- object-count consistency
- person-count consistency
- normalized object-center error

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
