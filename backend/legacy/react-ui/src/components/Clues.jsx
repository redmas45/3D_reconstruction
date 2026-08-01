/**
 * Entities YOLO tracked in the visible footage — the evidence the story is built on.
 *
 * Sorted by how long each was on screen, because a track seen for four seconds is worth
 * more to the reconstruction than one seen for four frames, and with well over a hundred
 * entities the useful ones must not be buried.
 */

const DIRECTION_ARROWS = {
  left: "←", right: "→", up: "↑", down: "↓",
  stationary: "•", "": "",
};

export default function Clues({ clues, timeline }) {
  const fps = timeline?.fps ?? 30;
  const entities = [...(clues?.entities ?? [])].sort(
    (a, b) => (b.frame_count ?? 0) - (a.frame_count ?? 0),
  );
  const classes = entities.reduce((counts, entity) => {
    counts[entity.class_name] = (counts[entity.class_name] ?? 0) + 1;
    return counts;
  }, {});

  return (
    <div className="panel">
      <h2>Clues — {entities.length} tracked entities</h2>
      {entities.length === 0 ? (
        <div className="empty">Waiting for detection and tracking…</div>
      ) : (
        <>
          <div className="row" style={{ marginBottom: 10 }}>
            {Object.entries(classes)
              .sort((a, b) => b[1] - a[1])
              .map(([name, count]) => (
                <span className="pill" key={name}>
                  <span className="dot" />
                  {name} × {count}
                </span>
              ))}
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Entity</th>
                  <th>Class</th>
                  <th>On screen</th>
                  <th>Frames</th>
                  <th>Moving</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {entities.map((entity) => (
                  <tr key={entity.id}>
                    <td><code>{entity.id}</code></td>
                    <td>{entity.class_name}</td>
                    <td className="muted">
                      {entity.first_frame != null && entity.last_frame != null
                        ? `${(entity.first_frame / fps).toFixed(1)}–${(entity.last_frame / fps).toFixed(1)}s`
                        : "—"}
                    </td>
                    <td className="muted">{entity.frame_count ?? "—"}</td>
                    <td className="muted">
                      {entity.direction
                        ? `${DIRECTION_ARROWS[entity.direction] ?? ""} ${entity.direction}`
                        : "—"}
                    </td>
                    <td className="muted">
                      {entity.confidence != null
                        ? `${(entity.confidence * 100).toFixed(0)}%`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
