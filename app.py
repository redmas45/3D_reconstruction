import argparse
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


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local forensic reconstruction interface")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the interface automatically")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output_root = ROOT / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(output_root / "server.log", encoding="utf-8"), logging.StreamHandler()],
    )
    manager = JobManager(
        upload_root=ROOT / "data" / "uploads",
        output_root=output_root / "jobs",
        config_data=load_config(),
        legacy_output_root=output_root,
    )
    server = build_server((args.host, args.port), manager, ROOT / "web")
    interface_url = f"http://{args.host}:{server.server_port}"
    print(f"\nForensic Reconstruction UI: {interface_url}")
    print("Press Ctrl+C to stop the local server.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(interface_url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local server...")
    finally:
        server.server_close()
        manager.shutdown()


if __name__ == "__main__":
    main()
