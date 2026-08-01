// @ts-check

const THEME_PREFERENCE_KEY = "reconstruct-theme";
const savedThemePreference = window.localStorage.getItem(THEME_PREFERENCE_KEY);
const systemPrefersLightTheme = window.matchMedia("(prefers-color-scheme: light)").matches;
const initialTheme = savedThemePreference === "dark" || savedThemePreference === "light"
  ? savedThemePreference
  : systemPrefersLightTheme ? "light" : "dark";

document.documentElement.dataset.theme = initialTheme;
