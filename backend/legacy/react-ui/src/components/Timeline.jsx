/**
 * The 75/25 split, drawn to scale.
 *
 * This is the clearest statement the interface makes: green is footage the system was
 * allowed to see, red is what was hidden from it and had to be reconstructed. Gaps turn
 * green as their render completes, so the bar doubles as render progress.
 */

function formatSeconds(value) {
  if (value == null) return "—";
  const minutes = Math.floor(value / 60);
  const seconds = (value % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

export default function Timeline({ timeline, render }) {
  if (!timeline) {
    return (
      <div className="panel">
        <h2>Timeline</h2>
        <div className="empty">Waiting for gap selection…</div>
      </div>
    );
  }

  const total = timeline.frame_count || 1;
  const completed = new Set(
    (render?.gaps ?? []).filter((gap) => gap.completed).map((gap) => gap.gap_index),
  );
  const asPercent = (frames) => `${(frames / total) * 100}%`;

  return (
    <div className="panel">
      <h2>
        Timeline — {timeline.hidden_ranges.length} gaps,{" "}
        {((timeline.missing_fraction ?? 0) * 100).toFixed(1)}% hidden
      </h2>

      <div className="timeline">
        {timeline.visible_ranges.map((range) => (
          <div
            key={`v-${range.start_frame}`}
            className="seg visible"
            style={{
              left: asPercent(range.start_frame),
              width: asPercent(range.end_frame - range.start_frame + 1),
            }}
            title={`Evidence ${formatSeconds(range.start_seconds)}–${formatSeconds(range.end_seconds)}`}
          />
        ))}
        {timeline.hidden_ranges.map((range) => (
          <div
            key={`g-${range.gap_index}`}
            className={`seg gap${completed.has(range.gap_index) ? " done" : ""}`}
            style={{
              left: asPercent(range.start_frame),
              width: asPercent(range.end_frame - range.start_frame + 1),
            }}
            title={
              `Gap ${range.gap_index + 1}: ${range.duration_seconds}s ` +
              `(${formatSeconds(range.start_seconds)}–${formatSeconds(range.end_seconds)})` +
              (completed.has(range.gap_index) ? " — reconstructed" : " — pending")
            }
          >
            <span className="label">{range.gap_index + 1}</span>
          </div>
        ))}
      </div>

      <div className="ruler">
        <span>0:00.0</span>
        <span>{formatSeconds(timeline.duration_seconds)}</span>
      </div>

      <div className="legend">
        <span><i style={{ background: "var(--visible)" }} />Evidence (analysed)</span>
        <span><i style={{ background: "var(--gap)" }} />Hidden (to reconstruct)</span>
        <span><i style={{ background: "#14532d" }} />Reconstructed</span>
        <span className="muted">
          {timeline.frame_count} frames @ {timeline.fps} fps
        </span>
      </div>
    </div>
  );
}
