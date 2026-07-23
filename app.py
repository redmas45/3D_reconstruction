import argparse
import ipaddress
import logging
import sys
import threading
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from application.processing_jobs import JobManager
from application.reconstruction_pipeline import load_config
from interfaces.http.local_server import build_server
from infrastructure.application_lock import (
    ApplicationAlreadyRunningError,
    ApplicationInstanceLock,
)
from infrastructure.environment import load_environment_file


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOCALHOST_NAMES = frozenset({"localhost"})


def _loopback_host(value: str) -> str:
    if value.lower() in LOCALHOST_NAMES:
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Host must be a loopback IP address or localhost") from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("This unauthenticated interface can bind only to loopback")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local forensic reconstruction interface")
    parser.add_argument("--host", default=DEFAULT_HOST, type=_loopback_host)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the interface automatically")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    load_environment_file(ROOT / ".env")
    output_root = ROOT / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    instance_lock = ApplicationInstanceLock(output_root / "server.lock")
    try:
        instance_lock.acquire()
    except ApplicationAlreadyRunningError as error:
        raise SystemExit(str(error)) from error
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(output_root / "server.log", encoding="utf-8"), logging.StreamHandler()],
    )
    manager: JobManager | None = None
    server = None
    try:
        manager = JobManager(
            upload_root=ROOT / "data" / "uploads",
            output_root=output_root / "jobs",
            config_data=load_config(),
            legacy_output_root=output_root,
        )
        server = build_server((args.host, args.port), manager, ROOT / "web")
        display_host = f"[{args.host}]" if ":" in args.host else args.host
        interface_url = f"http://{display_host}:{server.server_port}"
        print(f"\nForensic Reconstruction UI: {interface_url}")
        print("Press Ctrl+C to stop the local server.\n")
        if not args.no_browser:
            threading.Timer(0.6, lambda: webbrowser.open(interface_url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping local server...")
    finally:
        if server is not None:
            server.cancel_active_requests()
            server.server_close()
        if manager is not None:
            manager.shutdown()
        instance_lock.release()


if __name__ == "__main__":
    main()
