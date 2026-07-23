"""Serves one completed result through a loopback-only Colab proxy."""

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


VIDEO_CHUNK_BYTES = 1024 * 1024


class ResultViewerServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        video_path: Path,
        manifest_path: Path,
        web_root: Path,
    ) -> None:
        super().__init__(address, ResultViewerHandler)
        self.video_path = video_path.resolve()
        self.manifest_path = manifest_path.resolve()
        self.web_root = web_root.resolve()


class ResultViewerHandler(BaseHTTPRequestHandler):
    server: ResultViewerServer

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/":
            self._serve_static(self.server.web_root / "result.html")
            return
        if request_path == "/api/presentation":
            self._serve_manifest()
            return
        if request_path == "/api/video":
            self._serve_video()
            return
        if request_path.startswith("/assets/"):
            self._serve_asset(request_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_manifest(self) -> None:
        try:
            payload = json.loads(self.server.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(encoded, "application/json; charset=utf-8")

    def _serve_video(self) -> None:
        try:
            file_size = self.server.video_path.stat().st_size
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        byte_range = _parse_range(self.headers.get("Range"), file_size)
        if byte_range is None:
            start_byte, end_byte, status = 0, file_size - 1, HTTPStatus.OK
        else:
            start_byte, end_byte, status = *byte_range, HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end_byte - start_byte + 1))
        if status is HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start_byte}-{end_byte}/{file_size}")
        self.end_headers()
        self._stream_file(self.server.video_path, start_byte, end_byte)

    def _serve_asset(self, request_path: str) -> None:
        relative_path = Path(request_path.removeprefix("/"))
        candidate = (self.server.web_root / relative_path).resolve()
        if self.server.web_root not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._serve_static(candidate)

    def _serve_static(self, path: Path) -> None:
        try:
            content = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type = f"{content_type}; charset=utf-8"
        self._send_bytes(content, content_type)

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _stream_file(self, path: Path, start_byte: int, end_byte: int) -> None:
        remaining = end_byte - start_byte + 1
        try:
            with path.open("rb") as video_file:
                video_file.seek(start_byte)
                while remaining > 0:
                    chunk = video_file.read(min(VIDEO_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, message_format: str, *arguments: object) -> None:
        return


def start_result_viewer(
    video_path: Path,
    manifest_path: Path,
    web_root: Path,
) -> tuple[ResultViewerServer, threading.Thread]:
    for required_path in (video_path, manifest_path, web_root / "result.html"):
        if not required_path.exists():
            raise FileNotFoundError(f"Result viewer resource is missing: {required_path}")
    server = ResultViewerServer(
        ("127.0.0.1", 0), video_path, manifest_path, web_root,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="colab-result-viewer",
        daemon=True,
    )
    thread.start()
    return server, thread


def _parse_range(value: str | None, file_size: int) -> tuple[int, int] | None:
    if not value:
        return None
    unit, separator, range_value = value.partition("=")
    start_text, dash, end_text = range_value.partition("-")
    if unit != "bytes" or not separator or not dash or not start_text:
        return None
    try:
        start_byte = int(start_text)
        end_byte = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    if start_byte < 0 or start_byte >= file_size or end_byte < start_byte:
        return None
    return start_byte, min(end_byte, file_size - 1)
