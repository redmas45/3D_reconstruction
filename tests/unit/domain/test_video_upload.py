import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.video_upload import UploadValidationError, sanitize_upload_filename, validate_upload_metadata


class VideoUploadPolicyTests(unittest.TestCase):
    def test_sanitizes_path_and_unsafe_characters(self) -> None:
        self.assertEqual("judge_script_video.mp4", sanitize_upload_filename("../../judge<script>video?.MP4"))

    def test_accepts_supported_video_extension(self) -> None:
        self.assertEqual("evidence.webm", validate_upload_metadata("evidence.webm", 500, 1_000))

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaisesRegex(UploadValidationError, "Unsupported video type"):
            validate_upload_metadata("evidence.txt", 500, 1_000)

    def test_rejects_upload_over_size_limit(self) -> None:
        with self.assertRaisesRegex(UploadValidationError, "upload limit"):
            validate_upload_metadata("evidence.mp4", 1_001, 1_000)


if __name__ == "__main__":
    unittest.main()
