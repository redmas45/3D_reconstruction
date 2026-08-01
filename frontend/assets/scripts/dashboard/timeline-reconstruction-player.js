// @ts-check

import { ReconstructionView } from "../three/three-reconstruction-view.js";
import {
  compositeExportSupported,
  exportCompositeTimeline,
} from "./timeline-composite-export.js";

const MAXIMUM_VISIBLE_STORY_POINTS = 3;
const MAXIMUM_VISIBLE_CLUES = 3;
const MAXIMUM_VISIBLE_GAP_DETAILS = 12;
const GAP_PRELOAD_SECONDS = 1.25;
const DEFAULT_TIMELINE_STATUS =
  "Original visible frames are preserved. Sky-blue intervals are the inferred 3D replacement.";

/**
 * Builds the one thing the judge should see first: a normal timeline player. The
 * uploaded source remains underneath it, while a transparent Three.js layer appears
 * only inside the selected inferred intervals.
 *
 * @param {any} manifest
 * @param {string} videoUrl
 * @param {string} sourceName
 * @returns {{ element: HTMLElement, dispose: () => void }}
 */
export function createReconstructionPlaybackCard(manifest, videoUrl, sourceName) {
  const root = el("div", "timeline-reconstruction-card");
  const heading = el("header", "timeline-card-heading");
  const title = el("div");
  title.append(
    el("span", "timeline-card-kicker", "Reconstructed playback"),
    el("h3", "", `${stripExtension(sourceName)} · one continuous video`),
  );
  heading.append(title, el("span", "timeline-ready", "Ready to review"));

  const media = el("div", "timeline-media");
  const video = /** @type {HTMLVideoElement} */ (document.createElement("video"));
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.src = videoUrl;
  const overlay = el("div", "timeline-overlay");
  media.append(video, overlay);

  const status = el(
    "p",
    "timeline-status",
    DEFAULT_TIMELINE_STATUS,
  );
  const markers = buildTimelineMarkers(manifest, video);
  const brief = buildPlainLanguageBrief(manifest);
  root.append(heading, media, status, markers, brief);

  let view = null;
  let activeGap = -1;
  let preparedGap = -1;
  let animationFrame = 0;

  try {
    view = new ReconstructionView(overlay, { overlay: true, showUncertainty: false });
    view.load(manifest);
    view.pause();
  } catch (error) {
    overlay.replaceChildren(el("p", "timeline-overlay-fallback", "3D overlay could not be loaded."));
  }

  const sync = () => {
    const currentTime = Number(video.currentTime) || 0;
    const nextGapIndex = findUpcomingGap(manifest, currentTime, GAP_PRELOAD_SECONDS);
    if (nextGapIndex >= 0 && nextGapIndex !== preparedGap && view) {
      // Build the lightweight scene while the source footage is still visible. The
      // overlay is hidden during this call, so scene construction cannot freeze the
      // exact frame where the inferred interval begins.
      view.selectGap(nextGapIndex);
      preparedGap = nextGapIndex;
    }

    const gapIndex = findGap(manifest, currentTime);
    if (gapIndex < 0) {
      if (activeGap >= 0) status.textContent = DEFAULT_TIMELINE_STATUS;
      activeGap = -1;
      overlay.classList.remove("is-active");
      return;
    }
    const gap = manifest.gaps[gapIndex];
    if (gapIndex !== activeGap && view) {
      if (gapIndex !== preparedGap) {
        view.selectGap(gapIndex);
        preparedGap = gapIndex;
      }
    }
    if (gapIndex !== activeGap) {
      activeGap = gapIndex;
      status.textContent = `Inferred interval ${gapIndex + 1} of ${(manifest.gaps || []).length} is playing.`;
    }
    overlay.classList.add("is-active");
    if (view) view.setTimelineTime(currentTime - Number(gap.start_seconds || 0));
    markers.querySelectorAll("button").forEach((button, index) => {
      button.classList.toggle("active", index === gapIndex);
    });
  };

  const animateOverlay = () => {
    sync();
    if (!video.paused && !video.ended) animationFrame = window.requestAnimationFrame(animateOverlay);
  };
  const startOverlay = () => {
    if (!animationFrame) animationFrame = window.requestAnimationFrame(animateOverlay);
  };
  const stopOverlay = () => {
    if (animationFrame) window.cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    sync();
  };
  video.addEventListener("timeupdate", sync);
  video.addEventListener("seeked", sync);
  video.addEventListener("play", startOverlay);
  video.addEventListener("pause", stopOverlay);
  video.addEventListener("ended", stopOverlay);
  sync();

  const exportButton = buildExportButton(video, overlay, manifest, sourceName, sync, status);
  root.append(exportButton);

  return {
    element: root,
    dispose() {
      stopOverlay();
      video.removeEventListener("timeupdate", sync);
      video.removeEventListener("seeked", sync);
      video.removeEventListener("play", startOverlay);
      video.removeEventListener("pause", stopOverlay);
      video.removeEventListener("ended", stopOverlay);
      if (view) view.dispose();
      view = null;
    },
  };
}

function buildExportButton(video, overlay, manifest, sourceName, syncOverlay, status) {
  const button = /** @type {HTMLButtonElement} */ (el("button", "timeline-export-button", "Export review video"));
  button.type = "button";
  button.title = "Create a browser composite with the 3D layer visible in the inferred intervals.";
  if (!compositeExportSupported(video)) {
    button.disabled = true;
    button.title = "This browser cannot export a composite video; use the timeline player instead.";
    button.textContent = "Export unavailable";
    return button;
  }
  if (!video.videoWidth) {
    button.disabled = true;
    button.textContent = "Preparing export";
    video.addEventListener("loadedmetadata", () => {
      button.disabled = false;
      button.textContent = "Export review video";
    }, { once: true });
  }
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Exporting review video…";
    status.textContent = "Playing the timeline in real time to combine the video and 3D layer.";
    try {
      await exportCompositeTimeline(
        video,
        overlay,
        Number(manifest.source?.fps) || 30,
        `${stripExtension(sourceName)}_review_composite.webm`,
        syncOverlay,
      );
      status.textContent = "Composite review video downloaded. It includes the animated 3D intervals.";
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Composite export failed.";
    } finally {
      button.disabled = false;
      button.textContent = "Export review video";
    }
  });
  return button;
}

function buildTimelineMarkers(manifest, video) {
  const track = el("div", "timeline-track");
  track.setAttribute("aria-label", "Video evidence timeline");
  const duration = Math.max(0.001, Number(manifest.source?.duration_seconds) || 1);
  (manifest.gaps || []).forEach((gap, index) => {
    const marker = /** @type {HTMLButtonElement} */ (el("button", "timeline-gap-marker"));
    marker.type = "button";
    marker.style.left = `${(Number(gap.start_seconds) / duration) * 100}%`;
    marker.style.width = `${Math.max(0.8, (Number(gap.duration_seconds) / duration) * 100)}%`;
    marker.title = `Play inferred interval ${index + 1}`;
    marker.setAttribute("aria-label", marker.title);
    marker.addEventListener("click", () => {
      video.currentTime = Number(gap.start_seconds) || 0;
      void video.play();
    });
    track.append(marker);
  });
  const legend = el("div", "timeline-legend");
  legend.append(
    legendItem("source", "Original footage"),
    legendItem("inferred", "3D inferred interval"),
  );
  const wrap = el("div", "timeline-track-wrap");
  wrap.append(track, legend);
  return wrap;
}

function buildPlainLanguageBrief(manifest) {
  const section = el("section", "plain-language-brief");
  const story = manifest.narrative || {};
  const source = manifest.source || {};
  const heading = el("div", "brief-heading");
  heading.append(
    el("h4", "", "What the system found"),
    el("span", "brief-confidence", `${Math.round((Number(story.confidence) || 0) * 100)}% supported`),
  );
  section.append(
    heading,
    el(
      "p",
      "brief-summary",
      story.whole_video_summary || "The visible footage was analyzed and the missing intervals were bounded from their neighboring frames.",
    ),
  );

  const storyPoints = Array.isArray(story.story_points)
    ? story.story_points.filter((point) => typeof point === "string").slice(0, MAXIMUM_VISIBLE_STORY_POINTS)
    : [];
  if (storyPoints.length) {
    const observations = el("div", "brief-observations");
    observations.append(el("h4", "", "Key observations"));
    const list = el("ul");
    storyPoints.forEach((point) => list.append(el("li", "", point)));
    observations.append(list);
    section.append(observations);
  }

  const metrics = el("div", "brief-metrics");
  const observedPercent = formatPercent(source.observed_fraction, 75);
  const inferredPercent = formatPercent(source.reconstructed_fraction, 25);
  metrics.append(
    briefMetric(`${observedPercent}%`, "source kept", "Visible footage remains unchanged."),
    briefMetric(`${inferredPercent}%`, "intervals inferred", `${(manifest.gaps || []).length} replacement gaps`),
    briefMetric("3D", "browser layer", "Shown only while a gap plays."),
  );
  section.append(metrics);

  const clues = (manifest.clues || []).slice(0, MAXIMUM_VISIBLE_CLUES);
  const details = /** @type {HTMLDetailsElement} */ (el("details", "brief-details"));
  details.append(el("summary", "See clues and gap decisions"));
  const content = el("div", "brief-details-content");
  content.append(el("h4", "Clues from the visible footage"));
  const clueList = el("ul");
  clues.forEach((clue) => clueList.append(el("li", "", clue.statement)));
  if (!clueList.childElementCount) clueList.append(el("li", "", "Boundary tracks and measured movement continuity."));
  content.append(clueList, el("h4", "Intervals filled"), buildGapSummaryList(manifest.gaps));
  content.append(el("h4", "How the patch was made"));
  content.append(el("p", "", "The system observed the visible frames, selected a bounded motion hypothesis, then placed a lightweight animated 3D continuation only inside each marked interval."));
  details.append(content);
  section.append(details);
  return section;
}

function buildGapSummaryList(gaps) {
  const list = el("ul", "brief-gap-list");
  (Array.isArray(gaps) ? gaps : []).slice(0, MAXIMUM_VISIBLE_GAP_DETAILS).forEach((gap, index) => {
    const duration = Number(gap.duration_seconds) || 0;
    const description = gap.narrative?.inside_inferred || gap.decision?.gap_summary || "Bounded continuation from the visible boundaries.";
    list.append(el("li", "", `Gap ${index + 1} · ${duration.toFixed(1)}s — ${description}`));
  });
  if (!list.childElementCount) list.append(el("li", "", "No inferred intervals were recorded."));
  return list;
}

function formatPercent(value, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.round(Math.max(0, Math.min(1, numeric)) * 100);
}

function briefMetric(value, label, detail) {
  const metric = el("div", "brief-metric");
  metric.append(el("strong", "", value), el("span", "", label), el("small", "", detail));
  return metric;
}

function legendItem(state, label) {
  const item = el("span", "timeline-legend-item");
  item.append(el("i", `timeline-swatch ${state}`), el("span", "", label));
  return item;
}

function findGap(manifest, time) {
  return (manifest.gaps || []).findIndex((gap) => {
    const start = Number(gap.start_seconds) || 0;
    const end = Number(gap.end_seconds) || start;
    return time >= start && time < end;
  });
}

function findUpcomingGap(manifest, time, preloadSeconds) {
  return (manifest.gaps || []).findIndex((gap) => {
    const start = Number(gap.start_seconds) || 0;
    const end = Number(gap.end_seconds) || start;
    return time >= Math.max(0, start - preloadSeconds) && time < start && end > start;
  });
}

function stripExtension(value) {
  return String(value || "Video").replace(/\.[^.]+$/, "");
}

function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (typeof text === "string") node.textContent = text;
  return node;
}
