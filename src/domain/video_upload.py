import re
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg", ".wmv",
})
DEFAULT_MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAXIMUM_UPLOAD_FILENAME_LENGTH = 180
UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]+")


class UploadValidationError(ValueError):
    pass


def validate_upload_metadata(source_name: str, content_length: int, maximum_bytes: int) -> str:
    safe_name = sanitize_upload_filename(source_name)
    if len(safe_name) > MAXIMUM_UPLOAD_FILENAME_LENGTH:
        raise UploadValidationError("The video filename is too long")
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise UploadValidationError(f"Unsupported video type. Supported: {supported}")
    if content_length < 1:
        raise UploadValidationError("The uploaded video is empty")
    if content_length > maximum_bytes:
        limit_gib = maximum_bytes / (1024 ** 3)
        raise UploadValidationError(f"Video exceeds the {limit_gib:.1f} GiB upload limit")
    return safe_name


def sanitize_upload_filename(source_name: str) -> str:
    source_basename = Path(source_name).name.strip()
    sanitized_stem = UNSAFE_FILENAME_CHARACTERS.sub("_", Path(source_basename).stem).strip(" ._")
    normalized_suffix = Path(source_basename).suffix.lower()
    return f"{sanitized_stem or 'uploaded_video'}{normalized_suffix}"
