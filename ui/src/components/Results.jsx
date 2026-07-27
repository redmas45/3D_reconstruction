/**
 * The output: the recovered plate, each reconstructed gap beside the footage it
 * replaced, and the finished video.
 *
 * The side-by-side is deliberate. The hidden footage is shown only here, after the
 * fact, and never reaches any reconstruction stage — it is how you judge the result,
 * not something the system was allowed to use.
 */

import { gapTruthUrl, gapVideoUrl, plateUrl, videoUrl } from "../api";

export default function Results({ jobId, job, timeline, render, hasPlate }) {
  const gaps = render?.gaps ?? [];
  const completed = gaps.filter((gap) => gap.completed);
  const isDone = job?.status === "completed";

  if (!hasPlate && completed.length === 0 && !isDone) {
    return (
      <div className="panel">
        <h2>Results</h2>
        <div className="empty">Nothing rendered yet.</div>
      </div>
    );
  }

  return (
    <>
      {isDone && (
        <div className="panel">
          <h2>Reconstructed video</h2>
          <video controls src={videoUrl(jobId)} style={{ width: "100%", borderRadius: 8, background: "#000" }} />
          <p className="muted" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
            Full duration, frame rate and audio of the source, with hidden intervals
            replaced by evidence-grounded reconstructions.
          </p>
        </div>
      )}

      {hasPlate && (
        <div className="panel">
          <h2>Recovered background plate</h2>
          <img
            src={plateUrl(jobId)}
            alt="Clean plate recovered from visible frames"
            style={{ width: "100%", borderRadius: 8, display: "block" }}
          />
          <p className="muted" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
            Real footage with every tracked person and vehicle masked out, recovered by
            a per-pixel temporal median over visible frames. Only actors are rendered;
            this background is photographic.
          </p>
        </div>
      )}

      {completed.length > 0 && (
        <div className="panel">
          <h2>Gaps — reconstruction vs hidden footage</h2>
          {completed.map((gap) => {
            const range = timeline?.hidden_ranges?.[gap.gap_index];
            const report = gap.report ?? {};
            return (
              <div className="gap-card" key={gap.gap_index}>
                <h3>Gap {gap.gap_index + 1}</h3>
                <div className="meta">
                  {range ? `${range.start_seconds}s – ${range.end_seconds}s · ` : ""}
                  {report.rendered_frames != null
                    ? `${report.rendered_frames} rendered + ${report.reused_frames} reused → ${report.source_frames} frames`
                    : `${gap.layer_count} layers`}
                  {report.resolution ? ` · ${report.resolution[0]}×${report.resolution[1]}` : ""}
                </div>
                <div className="grid-2">
                  <div>
                    <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                      Reconstruction
                    </div>
                    <video controls loop src={gapVideoUrl(jobId, gap.gap_index)} />
                  </div>
                  <div>
                    <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                      Hidden footage (comparison only)
                    </div>
                    <video controls loop src={gapTruthUrl(jobId, gap.gap_index)} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
