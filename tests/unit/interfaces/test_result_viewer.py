import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interfaces.http.result_viewer import start_result_viewer


class ResultViewerTests(unittest.TestCase):
    def test_serves_manifest_and_ranged_video_from_fixed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            web_root = temporary_root / "web"
            web_root.mkdir()
            (web_root / "result.html").write_text("<h1>Result</h1>", encoding="utf-8")
            video_path = temporary_root / "preview.mp4"
            video_path.write_bytes(b"0123456789")
            manifest_path = temporary_root / "presentation_manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": 1, "status": "completed"}),
                encoding="utf-8",
            )
            server, thread = start_result_viewer(
                video_path, manifest_path, web_root,
            )
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(f"{base_url}/api/presentation") as response:
                    manifest = json.loads(response.read())
                request = urllib.request.Request(
                    f"{base_url}/api/video", headers={"Range": "bytes=2-5"},
                )
                with urllib.request.urlopen(request) as response:
                    status = response.status
                    payload = response.read()
                    content_range = response.headers["Content-Range"]
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual(206, status)
        self.assertEqual(b"2345", payload)
        self.assertEqual("bytes 2-5/10", content_range)


if __name__ == "__main__":
    unittest.main()
