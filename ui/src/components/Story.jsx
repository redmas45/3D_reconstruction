/**
 * The narrative the reasoning stage produced, and the per-gap decisions behind it.
 *
 * The disclosure line is not decoration. Everything below it is inference from the
 * visible 75%, and the interface says so wherever the story is shown.
 */

function Narrative({ story }) {
  if (!story) return null;
  const paragraphs = []
    .concat(story.summary ?? [], story.narrative ?? [], story.headline ?? [])
    .filter((item) => typeof item === "string" && item.trim());
  if (paragraphs.length === 0) return null;
  return (
    <div className="story">
      {paragraphs.map((text, index) => (
        <p key={index}>{text}</p>
      ))}
    </div>
  );
}

export default function Story({ story }) {
  if (!story) {
    return (
      <div className="panel">
        <h2>Storyline</h2>
        <div className="empty">
          Waiting for the reasoning stage to summarise the timeline…
        </div>
      </div>
    );
  }

  const gaps = story.gaps ?? [];
  const clues = story.top_clues ?? [];

  return (
    <div className="panel">
      <h2>Storyline</h2>

      {story.disclosure && <p className="story disclosure">{story.disclosure}</p>}

      <Narrative story={story.story} />

      {story.method?.mode && (
        <p className="muted" style={{ fontSize: 12 }}>
          Planner: <code>{story.method.mode}</code>
          {story.method.model ? ` · ${story.method.model}` : ""}
          {/* Named explicitly: a deterministic fallback must never look like the
              model produced the story. */}
          {story.method.warning && ` — ${story.method.warning}`}
        </p>
      )}

      {clues.length > 0 && (
        <>
          <h2 style={{ marginTop: 18 }}>Key clues</h2>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {clues.slice(0, 8).map((clue, index) => (
              <li key={index} className="muted" style={{ marginBottom: 4 }}>
                {typeof clue === "string" ? clue : clue.summary ?? JSON.stringify(clue)}
              </li>
            ))}
          </ul>
        </>
      )}

      {gaps.length > 0 && (
        <>
          <h2 style={{ marginTop: 18 }}>Per-gap reasoning</h2>
          {gaps.map((gap, index) => (
            <div className="gap-card" key={index}>
              <h3>
                Gap {(gap.gap_index ?? index) + 1}
                {gap.duration_seconds ? ` · ${gap.duration_seconds}s` : ""}
              </h3>
              {gap.narrative && <div className="meta">{gap.narrative}</div>}
              {gap.entities?.length > 0 && (
                <div className="meta">
                  Reconstructed: {gap.entities.map((e) => e.id ?? e).join(", ")}
                </div>
              )}
              {gap.confidence != null && (
                <span className="pill">
                  <span className="dot" />
                  confidence {(gap.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
          ))}
        </>
      )}

      {story.source === "decision_trace" && (
        <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>
          Showing the raw decision trace; the full narrative is written when the run
          completes.
        </p>
      )}
    </div>
  );
}
