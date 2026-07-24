# AI-Inferred Evidence Visualization & 3D Forensic Reconstruction

A local and cloud pipeline for replacing a distributed 25% of any input video with evidence-grounded, stylized Blender 3D forensic inference views. The system extracts visible clues from the remaining 75% of the video, uses Azure OpenAI (`gpt-5.4-mini`) for structured evidence reasoning, compiles a renderer storyboard, and renders 3D gap reconstructions using headless Blender 4.5 LTS.

---

## 📐 System Architecture

```mermaid
flowchart TD
    A[Input Video] --> B{Source Admission & Preflight}
    B -->|Pass| C[Gap Selection: Hidden 25% / Visible 75%]
    C --> D[Visible 75% Analysis]
    
    subgraph Visible Evidence Processing
        D --> D1[YOLO Object Tracking - BoT-SORT]
        D --> D2[YOLO Pose Estimation - COCO-17 Joints]
        D --> D3[Identity Registry & Appearance Descriptor]
        D --> D4[Camera Motion & Height Estimation]
    end
    
    Visible Evidence Processing --> E[Clue Catalog & Entity Hypotheses]
    E --> F[Azure OpenAI Evidence Reasoner - gpt-5.4-mini]
    
    F -->|Validated Decisions| G[Local Decision Validator]
    F -->|Fallback on Error| G
    
    G --> H[Storyboard Compiler]
    H --> I[Runtime Budget Gate & Representative Benchmark]
    
    I -->|Approved| J[Headless Blender 4.5 LTS Cycles Renderer]
    J --> K[Sparse 8 FPS Layered Rendering]
    K --> L[FPS & Resolution Normalization]
    
    L --> M[FFmpeg Audio-Preserving Video Stitcher]
    M --> N[Reconstructed 100% Video]
    
    N --> O[Ground Truth Evaluator - SSIM / PSNR / Entity Counts]
```

---

## 🎯 Target Product Contract

For every supported video processed via the local Web UI or Google Colab notebook:

1. **Gap Allocation**: Randomly selects multiple non-overlapping hidden gaps totaling exactly **25%** of the input video duration at frame precision.
2. **Review Profiles**: Uses 5–7 second review gaps for videos $\ge$ 60 seconds, and 1–3 second compact gaps for shorter inputs.
3. **Visible Range Tracking**: The remaining **75%** visible evidence is processed with YOLO object detection (`yolo26m.pt`) and tracking (BoT-SORT) for people, vehicles, and carried objects. Visible people receive COCO-17 joint tracking via `yolo26n-pose.pt`.
4. **Visible-Only Evidence Contract**: Hidden gap frames are strictly isolated and cryptographically validated. The reasoner and planner have zero access to hidden ground truth during reconstruction planning.
5. **Evidence Ledger & Clues**: Builds a structured evidence ledger ($v2$) and clue catalog ($v1$) detailing track histories, appearances, velocities, camera motion, and boundary observations.
6. **Azure OpenAI Reasoning**: Passes structured visible evidence to Azure OpenAI (`gpt-5.4-mini` Structured Outputs) to evaluate candidate hypotheses, select entity paths, and produce an auditable whole-video narrative.
7. **Fallback Safety**: If Azure OpenAI is unconfigured or returns invalid schemas, a deterministic fallback planner generates validated gap decisions.
8. **3D Forensic Rendering**: Renders hidden gaps in headless **Blender 4.5 LTS** using Cycles, pinhole camera projection, horizon fitting, rigged humanoid NLA animations, class-aware vehicle silhouettes, and contact shadows.
9. **Timeline Assembly**: Stitches visible 75% video segments and rendered 3D gaps into a unified video matching original frame count, FPS, and source audio track.
10. **Post-Evaluation**: Evaluates 3D reconstructions against hidden ground truth only after rendering is complete, computing SSIM, PSNR, and entity tracking accuracy.

> ⚠️ **Non-Ground-Truth Disclosure**: Generated gap segments are explicitly watermarked and labeled: **AI-inferred evidence visualization — not ground truth**.

---

## 🧠 AI Evidence Reasoning

The pipeline uses Azure OpenAI as an **evidence-constrained reasoning planner**, not an unconstrained video generator.

- **Native HTTPS Adapter**: Built using Python's standard `urllib.request` library — zero external OpenAI SDK dependency required.
- **Model Deployment**: Configured for `gpt-5.4-mini` via Azure OpenAI Responses API (`store=false`, strict JSON schema).
- **Inputs**:
  - Compact track histories, lifecycle events, and boundary observations from visible 75%.
  - Entity appearance descriptors, velocity profiles, and camera calibration parameters.
  - Deterministic candidate motion paths and candidate hypothesis library.
  - Bounded visible keyframes and entity crop image manifests (max 4 gaps, 12 low-detail images per request).
- **Outputs**:
  - Structured per-gap decisions selecting validated entity hypotheses, action tokens (`walk`, `run`, `hold`, `exit`, `enter`), and event beats.
  - Whole-video presentation narrative: headline summary, chronological story points, confidence ratings, rejected alternatives, and unresolved unknowns.
- **Caching & Validation**: Digest-keyed caching by evidence hash, clue set, prompt, schema version, and deployment name. All outputs are strictly validated by local Python validators before reaching Blender.

---

## 🎬 3D Forensic Reconstruction Engine

Hidden gaps are rendered using headless **Blender 4.5 LTS**:

- **Pinhole Camera & Horizon Fitting**: Establishes a shared pinhole camera model derived from visible person geometry, horizon estimation, and ground plane mapping.
- **Rigged Humanoid Motion Library**: Supported people use `assets/animation/humanoid_motion_library.blend`. Armatures are driven by looped NLA actions (idle, walk, brisk walk, run) phase-aligned to visible YOLO pose gait cycles.
- **Vehicle Silhouettes**: Class-aware 3D meshes for cars, trucks, and buses featuring visible wheel spoke rotation and steering alignment.
- **Temporal Background Compositing**: For static-camera footage, foreground detections are masked out to create a clean background backplate. Boundary frames crossfade through the gap while inferred 3D actors composite cleanly over top.
- **Adaptive Render Profile**:
  - **Resolution**: Scaled long-edge target between 960px and 1280px.
  - **Sample Count**: 2 Cycles path-tracing samples with OptiX / CUDA acceleration.
  - **Frame Rate**: Renders at sparse 8 FPS reconstruction rate, then normalizes back to exact original video FPS and frame count.
- **Diagnostic Layers**: Extracts 6 diagnostic review passes at key pose frames: composite, environment backplate, 3D actors, contact shadows, depth map, and uncertainty overlay.
- **Runtime Budget Gate**: Evaluates gap complexity, predicts total render time, and enforces a hard **120-minute gate** before full rendering begins.

---

## 📁 Directory & Output Structure

```text
├── app.py                              # Local Web UI server entrypoint
├── requirements.txt                    # Pinned Python dependencies
├── config/
│   ├── reconstruction_config.json      # Master configuration settings
│   └── botsort_reid.yaml               # BoT-SORT object tracking config
├── src/
│   ├── application/                    # Processing job & pipeline orchestrators
│   │   ├── blender_pipeline.py         # Blender asset prep & process execution
│   │   ├── evidence_reasoning.py       # Azure OpenAI reasoning orchestrator
│   │   ├── processing_jobs.py          # Job queue & lifecycle manager
│   │   └── reconstruction_pipeline.py  # End-to-end pipeline driver
│   ├── domain/                         # Core domain logic & data models
│   │   ├── camera_calibration.py       # Horizon fitting & pinhole projection
│   │   ├── clue_catalog.py             # Structured evidence clue definitions
│   │   ├── gap_decisions.py            # Entity hypothesis selection & validation
│   │   ├── identity_registry.py        # Cross-gap entity tracking & descriptors
│   │   ├── presentation_manifest.py    # Schema-v3 UI presentation DTOs
│   │   └── render_runtime_budget.py    # 120-minute budget gate & representative gap
│   ├── infrastructure/                 # Low-level I/O & adapters
│   │   ├── azure_openai_reasoner.py    # Native HTTPS Azure OpenAI client
│   │   ├── blender_runner.py           # Subprocess launcher for headless Blender
│   │   └── media_tools.py              # FFmpeg/FFprobe wrapper
│   └── interfaces/http/                # REST API & SSE server
├── blender/                            # In-Blender Python execution scripts
│   ├── scene_builder.py                # Main Blender scene composition script
│   ├── human_builder.py                # Rigged human armature & NLA animation setup
│   ├── vehicle_builder.py              # Silhouette 3D vehicle generation
│   ├── render_gap.py                   # Cycles frame renderer & PNG manifest writer
│   └── render_passes.py                # Diagnostic layer extraction
├── web/                                # Frontend web interface
│   ├── index.html                      # Primary user dashboard
│   ├── result.html                     # Judge / Auditor presentation dashboard
│   └── assets/                         # CSS styling, JS controllers & API client
├── colab/
│   └── reconstruction.ipynb            # Google Colab GPU notebook
├── scripts/                            # Helper scripts (Mixamo library packager)
└── outputs/jobs/<job_id>/              # Job output directory
    ├── <video_name>_reconstructed.mp4  # Final stitched reconstruction video
    ├── job.json                        # Job status & metadata
    └── _work/<video_name>_<hash>/      # Intermediate artifacts
        ├── evidence/                   # Evidence ledger v2 & clue catalog v1
        ├── reasoning/                  # Gap hypotheses v2, decisions v2 & narrative
        ├── storyboard/                 # Render storyboard & runtime estimate
        ├── presentation_manifest.json  # Schema-v3 public UI presentation manifest
        ├── gaps/gap_XX/blender/        # Per-gap .blend files & MP4 renders
        └── diagnostic_report.json      # Ground truth evaluation metrics
```

---

## 🛠️ Quick Start & Installation

### Prerequisites

- **Python**: 3.12+
- **Blender**: 4.5 LTS on `PATH` or configured location
- **FFmpeg & FFprobe**: Installed and available on system `PATH`
- **GPU**: NVIDIA GPU with CUDA/OptiX support recommended for Cycles rendering

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/redmas45/3D_reconstruction.git
   cd 3D_reconstruction
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables** (Optional for Azure OpenAI):
   Create a `.env` file in the project root:
   ```env
   AZURE_OPENAI_API_KEY=your_azure_api_key
   AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/
   AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5.4-mini
   ```

4. **Launch the Local Interface**:
   ```bash
   python app.py
   ```
   Access the dashboard at `http://127.0.0.1:8000`.

### Useful Server Options

```bash
# Custom host and port
python app.py --host 127.0.0.1 --port 8080

# Run in headless mode without automatically opening browser
python app.py --no-browser
```

---

## ☁️ Google Colab GPU Execution

To run the complete pipeline on cloud GPUs:

1. Open `colab/reconstruction.ipynb` in Google Colab.
2. Select a GPU runtime (T4 or better).
3. Execute notebook cells in order. The notebook:
   - Automatically installs **Blender 4.5 LTS** and **FFmpeg**.
   - Verifies PyTorch CUDA and executes a fast Cycles GPU probe (OptiX/CUDA).
   - Reads Azure secrets securely without saving credentials to disk or Drive.
   - Saves completed gap checkpoints to Google Drive (`MyDrive/3D_Reconstruction`).
   - Benchmarks the heaviest representative gap and waits for visual approval.
   - Serves the judge presentation view (`result.html`) via Colab's authenticated loopback proxy.

---

## ⚙️ Configuration Reference

Main configuration parameters in `config/reconstruction_config.json`:

```json
{
  "yolo": {
    "model": "yolo26m.pt",
    "pose_enabled": true,
    "pose_model": "yolo26n-pose.pt",
    "pose_confidence": 0.3,
    "frame_stride": 8
  },
  "gap": {
    "missing_fraction": 0.25,
    "min_seconds": 5.0,
    "max_seconds": 7.0,
    "compact_min_seconds": 1.0,
    "compact_max_seconds": 3.0,
    "review_profile_min_video_seconds": 60.0
  },
  "reasoning": {
    "enabled": true,
    "planner_schema_version": 2,
    "reasoning_effort": "medium",
    "maximum_gaps_per_batch": 4,
    "maximum_images_per_batch": 12,
    "image_detail": "low"
  },
  "renderer": {
    "default_mode": "blender",
    "blender_version": "4.5 LTS",
    "production_scale_percent": 45,
    "target_fps": 8,
    "cycles_samples": 2,
    "minimum_render_long_edge": 960,
    "maximum_render_long_edge": 1280,
    "hybrid_static_backplate": true,
    "maximum_predicted_render_seconds": 7200
  }
}
```

---

## 🧪 Testing & Verification

Run the full unit test suite covering domain contracts, reasoning schemas, camera projection, asset pipelines, and interface servers:

```bash
python -m pytest
```

---

## 📊 Post-Render Ground Truth Evaluation

After all hidden 3D gaps have rendered, the evaluator compares the reconstructed segments against hidden ground-truth video:

- **SSIM & PSNR**: Structural similarity and peak signal-to-noise ratio against photographic frames.
- **Boundary Continuity**: Image similarity at gap entry ($t_{\text{start}}$) and exit ($t_{\text{end}}$).
- **Entity Consistency**: Object and person count fidelity.
- **Center Tracking Error**: Normalized 2D spatial error of object centroids.

> *Note: Evaluation metrics serve as internal technical diagnostics comparing stylized 3D graphics against real-world video, not as a claim of exact visual recovery.*
