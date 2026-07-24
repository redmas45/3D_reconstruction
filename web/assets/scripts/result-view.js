// @ts-check

import {
  createGapMarkerTrack,
  createPresentationDetails,
  createPresentationOverview,
} from "./presentation-view.js";

const THEME_STORAGE_KEY = "reconstruct-theme";

const video = /** @type {HTMLVideoElement} */ (
  document.querySelector("#result-video")
);
const markerRoot = document.querySelector("#marker-root");
const overviewRoot = document.querySelector("#overview-root");
const presentationRoot = document.querySelector("#presentation-root");
const themeToggle = /** @type {HTMLButtonElement|null} */ (
  document.querySelector("#theme-toggle")
);

if (!video || !markerRoot || !overviewRoot || !presentationRoot || !themeToggle) {
  throw new Error("The result viewer layout is incomplete");
}

initializeThemeToggle(themeToggle);

try {
  const response = await fetch("/api/presentation", { cache: "no-store" });
  if (!response.ok) throw new Error("The presentation manifest is unavailable");
  const presentation = await response.json();
  markerRoot.replaceChildren(createGapMarkerTrack(presentation, video));
  overviewRoot.replaceChildren(createPresentationOverview(presentation));
  presentationRoot.replaceChildren(createPresentationDetails(presentation, video));
} catch (error) {
  const message = error instanceof Error ? error.message : "Could not load result";
  const failure = document.createElement("p");
  failure.className = "loading failure";
  failure.textContent = message;
  overviewRoot.replaceChildren(failure);
  presentationRoot.replaceChildren();
}


function initializeThemeToggle(button) {
  updateThemeToggleLabel(button);
  button.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = nextTheme;
    window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    updateThemeToggleLabel(button);
  });
}


function updateThemeToggleLabel(button) {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  const label = `Switch to ${nextTheme} theme`;
  button.setAttribute("aria-label", label);
  button.title = label;
}
