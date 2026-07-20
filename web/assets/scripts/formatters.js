// @ts-check

const BYTES_PER_UNIT = 1024;
const SECONDS_PER_MINUTE = 60;
const SECONDS_PER_HOUR = 3600;
const BYTE_UNITS = Object.freeze(["KB", "MB", "GB", "TB"]);

/** @param {string} fileName @returns {string} */
export function extractFileExtension(fileName) {
  const nameParts = fileName.toLowerCase().split(".");
  return nameParts.length > 1 ? nameParts.pop() || "" : "";
}

/** @param {number|null} byteCount @returns {string} */
export function formatByteCount(byteCount) {
  if (byteCount == null) return "—";
  if (byteCount < BYTES_PER_UNIT) return `${byteCount} B`;
  let displaySize = byteCount;
  let unitIndex = -1;
  do {
    displaySize /= BYTES_PER_UNIT;
    unitIndex += 1;
  } while (displaySize >= BYTES_PER_UNIT && unitIndex < BYTE_UNITS.length - 1);
  return `${displaySize.toFixed(displaySize >= 10 ? 1 : 2)} ${BYTE_UNITS[unitIndex]}`;
}

/** @param {number|null} durationSeconds @returns {string} */
export function formatDuration(durationSeconds) {
  if (durationSeconds == null) return "Calculating…";
  const roundedSeconds = Math.max(0, Math.round(durationSeconds));
  const hours = Math.floor(roundedSeconds / SECONDS_PER_HOUR);
  const minutes = Math.floor((roundedSeconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE);
  const seconds = roundedSeconds % SECONDS_PER_MINUTE;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

/** @param {unknown} error @returns {string} */
export function errorMessage(error) {
  return error instanceof Error ? error.message : "An unexpected local error occurred";
}
