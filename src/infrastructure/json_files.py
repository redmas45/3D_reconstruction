"""Reads recoverable JSON caches and replaces them atomically."""

import json
import time
import uuid
from pathlib import Path


JSON_REPLACE_ATTEMPTS = 8
JSON_RETRY_BASE_SECONDS = 0.05


def read_json_file(path: Path) -> object | None:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
        _replace_json_file(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _replace_json_file(temporary_path: Path, destination_path: Path) -> None:
    for attempt_index in range(JSON_REPLACE_ATTEMPTS):
        try:
            temporary_path.replace(destination_path)
            return
        except PermissionError:
            if attempt_index == JSON_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(JSON_RETRY_BASE_SECONDS * (attempt_index + 1))
