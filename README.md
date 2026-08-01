# RECONSTRUCT

RECONSTRUCT is a browser-first evidence visualization for missing moments in video.
YOLO measures the visible 75%, Azure Grok turns validated clues into a bounded story and
decision trace, and Three.js renders the inferred intervals as an inspectable animated
scene. The result is an AI-inferred visualization, not recovered ground truth.

## Workflow

1. Upload one ordinary video through the local browser interface.
2. Select several 5–7 second hidden intervals whose total is about 25% of the timeline.
3. Analyze only the visible ranges with YOLO detection, tracking, and boundary pose clues.
4. Send the validated visible-only clue catalog and gap decisions to Azure Grok.
5. Rank bounded motion hypotheses with deterministic evidence and physics checks; the
   model can explain or rank candidates, but it cannot invent a path or override a hard
   constraint.
6. Validate the summary, evidence references, hypotheses, confidence, and unknowns.
7. Build an evidence-safe playback base: visible source frames stay in order and hidden
   frames use a quality-gated visible-only plate or a moving transition between the two
   visible boundary frames (never a stale held frame).
8. Play that complete timeline in the browser with a transparent Three.js layer that
   appears only inside the selected gaps.

Hidden frames never enter YOLO, Azure, planning, or the scene manifest. The completed
MP4 is a normal frame-complete timeline with source audio; the generated actors are a
browser-rendered overlay synchronized to the marked intervals. The player prepares each
lightweight gap scene before its boundary, so Three.js construction does not pause the
video. The result card can also record that playback as a browser-composited WebM for a
self-contained presentation clip. This keeps the server output honest and preserves the
original video everywhere it is observed.

### Actor animation pipeline

The browser does not draw people from disconnected primitives. It loads two rigged glTF
humanoids and a shared locomotion pack, clones each skeleton safely, and plays the
`Idle_A`, `Walk`, `Jog`, or `Sprint` clip through a Three.js `AnimationMixer`. Mocap root
translation and rotation are removed because the evidence-grounded Catmullâ€“Rom path owns
world movement and heading. A small foot-contact correction keeps the lowest foot on the
calibrated ground plane, while visible boundary bbox anchors adjust apparent actor scale.
Weak-confidence tracks intentionally use a simpler silhouette so visual detail does not
overstate certainty. Asset provenance and licenses are recorded in
`frontend/assets/models/ASSET_LICENSES.md`.

### Evidence-to-scene decision logic

The scene is not generated from the prose summary. For each tracked entity and gap,
`backend/domain/gap_hypotheses.py` constructs a small candidate set: boundary-consistent
motion, measured continuation, reduced motion, hold, enter/exit, supported turn, and a
conservative proxy. `backend/domain/hypothesis_scoring.py` scores every candidate from
visible boundary confidence, identity continuity, lifecycle, heading change, soft
post-boundary residual, and speed/acceleration/turn limits. The score and components are
saved in the hypothesis artifact and surfaced in the dashboard.

Azure Grok receives those candidates as a bounded decision problem. After its response,
`backend/domain/gap_decisions.py` runs the deterministic safety gate: candidates with
impossible lifecycle/turn assumptions are rejected, and a materially safer measured
candidate replaces an unsafe model choice. This makes the language model an evidence
interpreter and explanation layer rather than the source of geometry. The post-gap
position is a soft consistency signal only; it is never used to solve hidden ground
truth exactly.

This design follows the tracking-by-detection pattern described by [SORT](https://arxiv.org/abs/1602.00763),
the appearance-aware identity handling in [Deep SORT](https://arxiv.org/abs/1703.07402),
and the motion/appearance/camera-compensation principles in
[BoT-SORT](https://arxiv.org/abs/2206.14651). Its dynamic-feasibility and uncertainty
checks are deliberately conservative rather than a large learned forecasting model;
[Trajectron++](https://arxiv.org/abs/2001.03093) motivates that separation between
trajectory hypotheses and physical feasibility. Camera calibration remains scoped per
shot and uses RANSAC plane estimation consistent with [OpenCV's homography API](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

## Azure configuration

Put the current values in the local `.env` file (never commit it):

```dotenv
AZURE_GROK_API_KEY=...
AZURE_GROK_BASE_URL=https://<your-foundry-endpoint>/openai/v1
AZURE_GROK_CHAT_DEPLOYMENT=grok-4-20-reasoning
```

The adapter calls `/openai/v1/responses`, keeps the key server-side, and falls back to a
deterministic bounded result if Azure is unavailable or returns invalid evidence.

## Run locally

The short PowerShell command is:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:8000`. One process serves the API and frontend; Blender, Colab,
React, and Vite are not required by the active path.

## Repository layout

```text
app.py                         # single loopback entrypoint
.env                           # local secrets; ignored by Git
requirements.txt               # Python dependencies
pytest.ini                     # test discovery and markers
rules.md                       # project engineering rules
frontend/                      # only browser application
  index.html                   # upload, queue, and result shell
  assets/scripts/              # API, dashboard, and Three.js modules
  assets/vendor/three/         # pinned local Three.js module
backend/                       # only Python project tree
  application/                 # orchestration and jobs
  domain/                      # evidence, reasoning, and manifest rules
  infrastructure/              # Azure, YOLO, media, and storage adapters
  interfaces/                  # HTTP and compatibility boundaries
  config/                      # validated runtime policy
  models/                      # local YOLO weights; ignored by Git
  runtime/                     # uploads and generated outputs; ignored by Git
  tools/                       # maintenance and fixture commands
  tests/                       # unit and integration tests
  docs/                        # architecture and design notes
  legacy/                      # inactive Blender/Colab provenance only
```

Only `frontend/`, `backend/`, and the small root entry/config files are project folders;
runtime artifacts and caches stay inside ignored backend paths.

## Verification

```powershell
python -m pytest
node --check frontend/assets/scripts/three/three-reconstruction-view.js
node --check frontend/assets/scripts/dashboard/reconstruction-dashboard.js
python -m json.tool frontend/assets/fixtures/sample-scene.json > $null
```

The dashboard presents the whole-video story, visible clues, observed/inferred timeline,
decision trace, confidence, unknowns, and the animated Three.js gap review without asking
judges to read raw JSON files.
