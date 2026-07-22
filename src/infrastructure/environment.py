"""Loads local development environment values without exposing secrets."""

import os
import re
from pathlib import Path


ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_environment_file(environment_path: Path) -> set[str]:
    if not environment_path.is_file():
        return set()
    loaded_names: set[str] = set()
    with environment_path.open("r", encoding="utf-8") as environment_file:
        for line_number, raw_line in enumerate(environment_file, start=1):
            parsed_value = _parse_environment_line(raw_line, line_number)
            if parsed_value is None:
                continue
            name, value = parsed_value
            if name in os.environ:
                continue
            os.environ[name] = value
            loaded_names.add(name)
    return loaded_names


def _parse_environment_line(raw_line: str, line_number: int) -> tuple[str, str] | None:
    stripped_line = raw_line.strip()
    if not stripped_line or stripped_line.startswith("#"):
        return None
    assignment = stripped_line.removeprefix("export ").split("=", 1)
    if len(assignment) != 2:
        raise ValueError(f"Invalid environment assignment on line {line_number}")
    name, raw_value = assignment[0].strip(), assignment[1].strip()
    if ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"Invalid environment name on line {line_number}")
    return name, _unquote_value(raw_value, line_number)


def _unquote_value(raw_value: str, line_number: int) -> str:
    if not raw_value:
        return ""
    if raw_value[0] not in {'"', "'"}:
        return raw_value
    if len(raw_value) < 2 or raw_value[-1] != raw_value[0]:
        raise ValueError(f"Unterminated environment value on line {line_number}")
    return raw_value[1:-1]
