"""Exposes the local reconstruction API and dependency-free browser interface."""

import json
import ipaddress
import logging
import mimetypes
import os
import re
import socket
import stat
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, quote, unquote, urlparse, urlsplit

from application.processing_jobs import JobConflictError, JobManager, JobNotFoundError, UploadValidationError


LOGGER = logging.getLogger(__name__)
HEALTH_API_PATH = "/api/health"
JOBS_API_PATH = "/api/jobs"
STATIC_ASSET_PREFIX = "/assets/"
INDEX_REQUEST_PATHS = frozenset({"/", "/index.html"})
JOB_PATH_PATTERN = re.compile(r"^/api/jobs/([a-f0-9]{32})$")
JOB_CANCEL_PATH_PATTERN = re.compile(r"^/api/jobs/([a-f0-9]{32})/cancel$")
OUTPUT_PATH_PATTERN = re.compile(r"^/api/outputs/([a-f0-9]{32})$")
FILE_STREAM_CHUNK_BYTES = 1024 * 1024
REQUEST_SOCKET_TIMEOUT_SECONDS = 30
MAXIMUM_RANGE_DIGITS = 20
MAXIMUM_FAILED_UPLOAD_DRAIN_BYTES = 1024 * 1024
FAILED_UPLOAD_DRAIN_TIMEOUT_SECONDS = 2.0


class InvalidRangeError(ValueError):
    pass


class CountingRequestReader:
    def __init__(self, reader: BinaryIO) -> None:
        self._reader = reader
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        content = self._reader.read(size)
        self.bytes_read += len(content)
        return content


class ReconstructionHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address: tuple[str, int], manager: JobManager, web_root: Path) -> None:
        super().__init__(address, ReconstructionRequestHandler)
        self.manager = manager
        self.web_root = web_root.resolve()
        self.upload_semaphore = threading.BoundedSemaphore(value=1)
        self._active_connections: set[socket.socket] = set()
        self._active_connections_lock = threading.Lock()

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: tuple[object, ...],
    ) -> None:
        request.settimeout(REQUEST_SOCKET_TIMEOUT_SECONDS)
        with self._active_connections_lock:
            self._active_connections.add(request)
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._active_connections_lock:
                self._active_connections.discard(request)

    def cancel_active_requests(self) -> None:
        with self._active_connections_lock:
            connections = list(self._active_connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                continue

    def handle_error(self, request: socket.socket, client_address: tuple[object, ...]) -> None:
        connection_errors = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError)
        if isinstance(sys.exc_info()[1], connection_errors):
            return
        super().handle_error(request, client_address)


class IPv6ReconstructionHTTPServer(ReconstructionHTTPServer):
    address_family = socket.AF_INET6


class ReconstructionRequestHandler(BaseHTTPRequestHandler):
    server: ReconstructionHTTPServer

    def do_GET(self) -> None:
        if not self._validate_request_host():
            return
        request = urlparse(self.path)
        if request.path == HEALTH_API_PATH:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if request.path == JOBS_API_PATH:
            self._send_json(HTTPStatus.OK, {"jobs": self.server.manager.list_jobs()})
            return
        job_match = JOB_PATH_PATTERN.fullmatch(request.path)
        if job_match:
            self._get_job(job_match.group(1))
            return
        output_match = OUTPUT_PATH_PATTERN.fullmatch(request.path)
        if output_match:
            download = parse_qs(request.query).get("download") == ["1"]
            self._get_output(output_match.group(1), download)
            return
        self._get_static(request.path)

    def do_POST(self) -> None:
        if not self._validate_request_host():
            return
        request = urlparse(self.path)
        cancel_match = JOB_CANCEL_PATH_PATTERN.fullmatch(request.path)
        if cancel_match:
            self._cancel_job(cancel_match.group(1))
            return
        if request.path != JOBS_API_PATH:
            self._send_error(HTTPStatus.NOT_FOUND, "Endpoint was not found")
            return
        source_name = unquote(self.headers.get("X-File-Name", ""))
        renderer_mode = self.headers.get("X-Renderer-Mode", "blender")
        if not self.server.upload_semaphore.acquire(blocking=False):
            self._send_error(HTTPStatus.CONFLICT, "Another video upload is already in progress")
            return
        content_length = 0
        request_reader = CountingRequestReader(self.rfile)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            job = self.server.manager.create_job(source_name, request_reader, content_length, renderer_mode)
        except (UploadValidationError, ValueError) as error:
            self._drain_failed_upload(content_length - request_reader.bytes_read)
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        except OSError:
            LOGGER.exception("Could not save uploaded video %s", source_name)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "The upload could not be saved")
            return
        finally:
            self.server.upload_semaphore.release()
        self._send_json(HTTPStatus.ACCEPTED, {"job": job})

    def _drain_failed_upload(self, remaining_bytes: int) -> None:
        if remaining_bytes <= 0:
            return
        if remaining_bytes > MAXIMUM_FAILED_UPLOAD_DRAIN_BYTES:
            self.close_connection = True
            return
        deadline = time.monotonic() + FAILED_UPLOAD_DRAIN_TIMEOUT_SECONDS
        try:
            while remaining_bytes > 0:
                timeout_seconds = deadline - time.monotonic()
                if timeout_seconds <= 0.0:
                    self.close_connection = True
                    return
                self.connection.settimeout(timeout_seconds)
                content = self.rfile.read(min(64 * 1024, remaining_bytes))
                if not content:
                    break
                remaining_bytes -= len(content)
        except OSError:
            self.close_connection = True
        finally:
            if not self.close_connection:
                try:
                    self.connection.settimeout(REQUEST_SOCKET_TIMEOUT_SECONDS)
                except OSError:
                    self.close_connection = True

    def _cancel_job(self, job_id: str) -> None:
        try:
            job = self.server.manager.cancel_job(job_id)
        except JobNotFoundError as error:
            self._send_error(HTTPStatus.NOT_FOUND, str(error))
            return
        except JobConflictError as error:
            self._send_error(HTTPStatus.CONFLICT, str(error))
            return
        except OSError:
            LOGGER.exception("Could not persist cancellation for job %s", job_id)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "The cancellation could not be saved")
            return
        self._send_json(HTTPStatus.ACCEPTED, {"job": job})

    def do_DELETE(self) -> None:
        if not self._validate_request_host():
            return
        request = urlparse(self.path)
        job_match = JOB_PATH_PATTERN.fullmatch(request.path)
        if not job_match:
            self._send_error(HTTPStatus.NOT_FOUND, "Endpoint was not found")
            return
        try:
            self.server.manager.delete_job(job_match.group(1))
        except JobNotFoundError as error:
            self._send_error(HTTPStatus.NOT_FOUND, str(error))
            return
        except JobConflictError as error:
            self._send_error(HTTPStatus.CONFLICT, str(error))
            return
        except OSError:
            LOGGER.exception("Could not delete job %s", job_match.group(1))
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "The output is in use or could not be deleted")
            return
        self._send_json(HTTPStatus.OK, {"deleted": True})

    def _get_job(self, job_id: str) -> None:
        try:
            job = self.server.manager.get_job(job_id)
        except JobNotFoundError as error:
            self._send_error(HTTPStatus.NOT_FOUND, str(error))
            return
        self._send_json(HTTPStatus.OK, {"job": job})

    def _get_output(self, job_id: str, download: bool) -> None:
        try:
            output_path = self.server.manager.output_path(job_id)
        except JobNotFoundError as error:
            self._send_error(HTTPStatus.NOT_FOUND, str(error))
            return
        except JobConflictError as error:
            self._send_error(HTTPStatus.CONFLICT, str(error))
            return
        self._serve_file(output_path, download=download, allow_range=True)

    def _get_static(self, request_path: str) -> None:
        static_path = self._resolve_static_path(request_path)
        if static_path is None:
            self._send_error(HTTPStatus.NOT_FOUND, "Page was not found")
            return
        self._serve_file(static_path, download=False, allow_range=False)

    def _resolve_static_path(self, request_path: str) -> Path | None:
        if request_path in INDEX_REQUEST_PATHS:
            return self.server.web_root / "index.html"
        if not request_path.startswith(STATIC_ASSET_PREFIX):
            return None
        relative_path = unquote(request_path).lstrip("/")
        candidate_path = (self.server.web_root / relative_path).resolve()
        if self.server.web_root not in candidate_path.parents:
            return None
        return candidate_path

    def _serve_file(self, path: Path, download: bool, allow_range: bool) -> None:
        try:
            source_file = path.open("rb")
        except OSError:
            self._send_error(HTTPStatus.NOT_FOUND, "File was not found")
            return
        with source_file:
            file_status = os.fstat(source_file.fileno())
            if not stat.S_ISREG(file_status.st_mode):
                self._send_error(HTTPStatus.NOT_FOUND, "File was not found")
                return
            file_size = file_status.st_size
            try:
                start, end, is_partial = self._resolve_file_range(file_size, allow_range)
            except InvalidRangeError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT if is_partial else HTTPStatus.OK
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_file_headers(
                status, path, content_type, file_size, start, end, allow_range, is_partial, download,
            )
            self._stream_file(source_file, start, end)

    def _resolve_file_range(self, file_size: int, allow_range: bool) -> tuple[int, int, bool]:
        if not allow_range:
            return 0, file_size - 1, False
        parsed_range = self._parse_range(file_size)
        if parsed_range is None:
            return 0, file_size - 1, False
        return parsed_range[0], parsed_range[1], True

    def _send_file_headers(
        self,
        status: HTTPStatus,
        path: Path,
        content_type: str,
        file_size: int,
        start: int,
        end: int,
        allow_range: bool,
        is_partial: bool,
        download: bool,
    ) -> None:
        self.send_response(status)
        self._send_security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        if allow_range:
            self.send_header("Accept-Ranges", "bytes")
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        if download:
            self.send_header("Content-Disposition", _download_content_disposition(path.name))
        self.end_headers()

    def _stream_file(self, source_file: BinaryIO, start: int, end: int) -> None:
        try:
            source_file.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = source_file.read(min(FILE_STREAM_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except OSError:
            return

    def _parse_range(self, file_size: int) -> tuple[int, int] | None:
        header = self.headers.get("Range")
        if not header:
            return None
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if match is None or file_size < 1:
            raise InvalidRangeError("Requested byte range is invalid")
        start_text, end_text = match.groups()
        if not start_text and not end_text:
            raise InvalidRangeError("Requested byte range is empty")
        if len(start_text) > MAXIMUM_RANGE_DIGITS or len(end_text) > MAXIMUM_RANGE_DIGITS:
            raise InvalidRangeError("Requested byte range is too large")
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length < 1:
                raise InvalidRangeError("Requested suffix range is invalid")
            return max(0, file_size - suffix_length), file_size - 1
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
        if start >= file_size or end < start:
            raise InvalidRangeError("Requested byte range is outside the file")
        return start, min(end, file_size - 1)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def _validate_request_host(self) -> bool:
        host_header = self.headers.get("Host", "")
        try:
            hostname = urlsplit(f"//{host_header}").hostname
        except ValueError:
            hostname = None
        if hostname is not None and _is_loopback_host(hostname):
            return True
        self._send_error(HTTPStatus.MISDIRECTED_REQUEST, "Request host is not allowed")
        return False

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self'; script-src 'self'; style-src 'self'")

    def log_message(self, message_format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), message_format % args)


def build_server(address: tuple[str, int], manager: JobManager, web_root: Path) -> ReconstructionHTTPServer:
    if not _is_loopback_host(address[0]):
        raise ValueError("The unauthenticated reconstruction server can bind only to loopback")
    if not (web_root / "index.html").is_file():
        raise FileNotFoundError(f"Web UI was not found: {web_root}")
    server_class = IPv6ReconstructionHTTPServer if ":" in address[0] else ReconstructionHTTPServer
    return server_class(address, manager, web_root)


def _is_loopback_host(hostname: str) -> bool:
    normalized_hostname = hostname.lower().rstrip(".")
    if normalized_hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized_hostname).is_loopback
    except ValueError:
        return False


def _download_content_disposition(filename: str) -> str:
    ascii_filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename).strip(".") or "reconstruction.mp4"
    encoded_filename = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{encoded_filename}"
