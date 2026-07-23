// @ts-check

import { createGapMarkerTrack, createPresentationView } from "./presentation-view.js";


const video = /** @type {HTMLVideoElement} */ (
  document.querySelector("#result-video")
);
const markerRoot = document.querySelector("#marker-root");
const presentationRoot = document.querySelector("#presentation-root");

if (!video || !markerRoot || !presentationRoot) {
  throw new Error("The result viewer layout is incomplete");
}

try {
  const response = await fetch("/api/presentation", { cache: "no-store" });
  if (!response.ok) throw new Error("The presentation manifest is unavailable");
  const presentation = await response.json();
  markerRoot.replaceChildren(createGapMarkerTrack(presentation, video));
  presentationRoot.replaceChildren(createPresentationView(presentation, video));
} catch (error) {
  const message = error instanceof Error ? error.message : "Could not load result";
  const failure = document.createElement("p");
  failure.className = "loading failure";
  failure.textContent = message;
  presentationRoot.replaceChildren(failure);
}
