import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.blender_preview import prepare_and_render_gap


DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "input_vid3.mp4"
DEFAULT_SCENE_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "preview_input_vid3" / "_work" / "input_vid3" / "scene_report.json"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "blender_preview_input_vid3"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one evidence-only gap with Blender")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--scene-report", type=Path, default=DEFAULT_SCENE_REPORT_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--gap-index", type=int, default=0)
    parser.add_argument("--mode", choices=("preview", "animation"), default="preview")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = prepare_and_render_gap(
        project_root=PROJECT_ROOT,
        video_path=arguments.video.resolve(),
        scene_report_path=arguments.scene_report.resolve(),
        gap_index=arguments.gap_index,
        output_directory=arguments.output_directory.resolve(),
        mode=arguments.mode,
    )
    print(f"Rendered media: {result.output_path}")
    print(f"Render report: {result.report_path}")
    print(f"Blender scene: {result.blend_path}")


if __name__ == "__main__":
    main()
