import { useCallback, useEffect, useRef, useState } from "react";

import { cancelJob, deleteJob, getHealth, listJobs, streamJob, uploadVideo } from "./api";
import Clues from "./components/Clues";
import Results from "./components/Results";
import Story from "./components/Story";
import Timeline from "./components/Timeline";

const STAGE_LABELS = {
  queued: "Queued",
  validating: "Checking tools and video",
  segmenting_shots: "Finding scene cuts",
  selecting_gaps: "Choosing hidden intervals",
  preparing: "Preparing segments",
  detecting: "Detecting and tracking (YOLO)",
  planning: "Building hypotheses",
  extracting_clues: "Writing the evidence ledger",
  reasoning: "Summarising into a storyline",
  validating_decisions: "Validating decisions",
  rendering: "Rendering and compositing",
  evaluating: "Evaluating against hidden truth",
  stitching: "Stitching the final video",
  completed: "Complete",
  failed: "Failed",
  cancelled: "Cancelled",
};

function Health({ health }) {
  if (!health) return null;
  const library = health.actor_library ?? {};
  const tool = (name, state) => (
    <span className={`pill ${state?.available ? "ok" : "bad"}`} key={name}>
      <span className="dot" />
      {name}
      {!state?.available && " missing"}
    </span>
  );
  return (
    <div className="row" style={{ marginTop: 10 }}>
      {tool("Blender", health.blender)}
      {tool("FFmpeg", health.ffmpeg)}
      {tool("FFprobe", health.ffprobe)}
      <span className={`pill ${library.available ? "ok" : "warn"}`}>
        <span className="dot" />
        {library.available
          ? `Prebuilt models: ${library.asset_count}`
          : `Prebuilt models ${library.reason ?? "unavailable"} — generating at runtime`}
      </span>
      <span className="pill">
        <span className="dot" />
        {health.renderable_classes?.length ?? 0} renderable classes
      </span>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [state, setState] = useState({});
  const [uploadFraction, setUploadFraction] = useState(null);
  const [error, setError] = useState(null);
  const unsubscribe = useRef(null);
  const fileInput = useRef(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((problem) => setError(problem.message));
    listJobs()
      .then((payload) => setJobs(payload.jobs))
      .catch((problem) => setError(problem.message));
  }, []);

  const watch = useCallback((jobId) => {
    if (unsubscribe.current) unsubscribe.current();
    setActiveId(jobId);
    setState({});
    unsubscribe.current = streamJob(
      jobId,
      (payload) => setState(payload),
      () => listJobs().then((p) => setJobs(p.jobs)).catch(() => {}),
    );
  }, []);

  useEffect(() => () => unsubscribe.current?.(), []);

  async function onUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    setUploadFraction(0);
    try {
      const job = await uploadVideo(file, setUploadFraction);
      setJobs((current) => [job, ...current]);
      watch(job.id);
    } catch (problem) {
      setError(problem.message);
    } finally {
      setUploadFraction(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const job = state.job ?? jobs.find((item) => item.id === activeId);
  const running = job && !["completed", "failed", "cancelled"].includes(job.status);

  return (
    <div className="app">
      <header className="masthead">
        <h1>AI-Inferred Forensic Reconstruction</h1>
        <p>
          Hides 25% of a video, analyses only what remains, and reconstructs the missing
          intervals as evidence-grounded actors composited onto the real scene.
        </p>
        <Health health={health} />
      </header>

      {error && <div className="error" style={{ marginBottom: 16 }}>{error}</div>}

      <div className="panel">
        <h2>Source video</h2>
        <div className="row">
          <input ref={fileInput} type="file" accept="video/*" onChange={onUpload} disabled={uploadFraction !== null} />
          {uploadFraction !== null && (
            <span className="muted">Uploading {(uploadFraction * 100).toFixed(0)}%</span>
          )}
          {running && (
            <button className="danger" onClick={() => cancelJob(job.id).catch((p) => setError(p.message))}>
              Cancel run
            </button>
          )}
        </div>
        {jobs.length > 0 && (
          <div className="row" style={{ marginTop: 12 }}>
            {jobs.slice(0, 6).map((item) => (
              <button
                key={item.id}
                className="ghost"
                style={{
                  borderColor: item.id === activeId ? "var(--accent)" : "var(--line)",
                }}
                onClick={() => watch(item.id)}
              >
                {item.source_name} · {item.status}
              </button>
            ))}
          </div>
        )}
      </div>

      {job && (
        <div className="panel">
          <div className="stage-line">
            <strong>{STAGE_LABELS[job.stage] ?? job.stage}</strong>
            <span className="detail">{job.detail}</span>
          </div>
          <div className="bar">
            <span style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
          </div>
          <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
            <span className="muted">{Math.round((job.progress ?? 0) * 100)}%</span>
            {job.eta_seconds != null && (
              <span className="muted">~{Math.round(job.eta_seconds / 60)} min remaining</span>
            )}
          </div>
          {job.error && <div className="error" style={{ marginTop: 12 }}>{job.error}</div>}
        </div>
      )}

      {job && (
        <>
          <Timeline timeline={state.timeline} render={state.render} />
          <div className="grid-2">
            <Clues clues={state.clues} timeline={state.timeline} />
            <Story story={state.story} />
          </div>
          <Results
            jobId={job.id}
            job={job}
            timeline={state.timeline}
            render={state.render}
            hasPlate={state.has_plate}
          />
          {job.status !== "queued" && !running && (
            <button
              className="ghost"
              onClick={() =>
                deleteJob(job.id)
                  .then(() => listJobs())
                  .then((payload) => {
                    setJobs(payload.jobs);
                    setActiveId(null);
                    setState({});
                  })
                  .catch((problem) => setError(problem.message))
              }
            >
              Delete this run
            </button>
          )}
        </>
      )}

      {!job && (
        <div className="panel">
          <div className="empty">Upload a video to begin.</div>
        </div>
      )}
    </div>
  );
}
