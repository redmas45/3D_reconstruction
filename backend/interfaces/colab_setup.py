"""Colab runtime setup, owned by the repository rather than the notebook.

Standard library only, deliberately: this runs immediately after the repo is cloned
and before `pip install -r requirements.txt`, so it cannot depend on numpy, cv2, or
anything else from requirements.

Pairs with `colab_run`, which owns the reconstruction itself. Between them the
notebook is reduced to Colab-specific glue: mounting Drive, two upload widgets, and
displaying results.
"""

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BLENDER_VERSION = "4.5.10"
BLENDER_ARCHIVE_URL_TEMPLATE = (
    "https://download.blender.org/release/Blender4.5/blender-{version}-linux-x64.tar.xz"
)
BLENDER_EXECUTABLE_ENVIRONMENT_KEY = "BLENDER_EXECUTABLE"

SYSTEM_PACKAGES = ("ffmpeg", "xvfb", "xauth", "libgl1", "libxi6", "libxrender1")

CYCLES_DEVICE_PREFERENCE = ("OPTIX", "CUDA")
CYCLES_PROBE_TIMEOUT_SECONDS = 180
GRAPHICS_PROBE_TIMEOUT_SECONDS = 60
CYCLES_READY_MARKER = "CYCLES_GPU_READY"

WORKBENCH_ENGINE = "BLENDER_WORKBENCH"
EEVEE_ENGINE = "BLENDER_EEVEE_NEXT"
CYCLES_ENGINE = "CYCLES"


class ColabSetupError(RuntimeError):
    """Setup could not produce a usable runtime."""


@dataclass(frozen=True)
class RenderCapability:
    """What this Colab machine can actually render with."""

    engine: str
    compute_device: str
    parallel_gap_renders: int
    detail: str

    @property
    def uses_gpu(self) -> bool:
        return self.engine == CYCLES_ENGINE


def require_nvidia_gpu() -> str:
    if shutil.which("nvidia-smi") is None:
        raise ColabSetupError(
            "No NVIDIA GPU was assigned. Set Runtime > Change runtime type > GPU, then reconnect."
        )
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return result.stdout


def install_python_requirements(project_root: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=str(project_root),
        check=True,
    )


def install_system_packages() -> None:
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", *SYSTEM_PACKAGES], check=True)


def install_blender(install_root: Path, version: str = BLENDER_VERSION) -> Path:
    """Download Blender and wrap it in xvfb so it has a display in a headless VM."""
    binary = install_root / f"blender-{version}-linux-x64" / "blender"
    if not binary.is_file():
        install_root.mkdir(parents=True, exist_ok=True)
        archive_path = install_root / "blender.tar.xz"
        subprocess.run(
            [
                "curl", "--fail", "--location",
                BLENDER_ARCHIVE_URL_TEMPLATE.format(version=version),
                "--output", str(archive_path),
            ],
            check=True,
        )
        subprocess.run(["tar", "-xf", str(archive_path), "-C", str(install_root)], check=True)
        archive_path.unlink(missing_ok=True)
    if not binary.is_file():
        raise ColabSetupError(f"Blender was not installed at {binary}")
    return _write_xvfb_wrapper(install_root, binary)


def _write_xvfb_wrapper(install_root: Path, binary: Path) -> Path:
    wrapper = install_root / "blender-colab"
    wrapper.write_text(
        "#!/usr/bin/env bash\nexec xvfb-run -a " + shlex.quote(str(binary)) + ' "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    # The rest of the codebase discovers Blender through this variable.
    os.environ[BLENDER_EXECUTABLE_ENVIRONMENT_KEY] = str(wrapper)
    return wrapper


def probe_cycles_device(blender_executable: Path, project_root: Path) -> RenderCapability:
    """Render a real frame on each candidate device rather than trusting a driver string.

    §5.6 and the M0 benchmark both make the same point: device availability has to be
    measured. A machine that reports a GPU can still fail to compile Cycles kernels.
    """
    probe_script = project_root / "blender" / "probe_cycles.py"
    output_path = project_root / "cycles_probe.png"
    for device in CYCLES_DEVICE_PREFERENCE:
        if _cycles_device_renders(blender_executable, probe_script, output_path, device):
            return RenderCapability(
                engine=CYCLES_ENGINE,
                compute_device=device,
                parallel_gap_renders=1,
                detail=f"Cycles {device} verified by a real render",
            )
    return RenderCapability(
        engine=WORKBENCH_ENGINE,
        compute_device="CPU",
        parallel_gap_renders=2,
        detail="Cycles GPU unavailable; falling back to Workbench on CPU",
    )


def _cycles_device_renders(
    blender_executable: Path, probe_script: Path, output_path: Path, device: str,
) -> bool:
    if not probe_script.is_file():
        return False
    command = [
        str(blender_executable), "--background", "--python", str(probe_script), "--",
        "--device", device, "--output", str(output_path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=CYCLES_PROBE_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    combined_output = result.stdout + result.stderr
    return (
        result.returncode == 0
        and CYCLES_READY_MARKER in combined_output
        and output_path.is_file()
    )


def report_graphics_platform(blender_executable: Path) -> str:
    """Blender's OpenGL vendor. Informational only — it does not predict Cycles."""
    expression = (
        "import bpy, gpu; "
        "print('VENDOR=' + gpu.platform.vendor_get()); "
        "print('RENDERER=' + gpu.platform.renderer_get()); "
        "bpy.ops.wm.quit_blender()"
    )
    try:
        result = subprocess.run(
            [str(blender_executable), "--factory-startup", "--python-expr", expression],
            capture_output=True, text=True,
            timeout=GRAPHICS_PROBE_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return "graphics probe unavailable"
    lines = [
        line for line in (result.stdout + result.stderr).splitlines()
        if line.startswith(("VENDOR=", "RENDERER="))
    ]
    return " ".join(lines) if lines else "graphics probe unavailable"


def verify_cuda_available() -> str:
    """PyTorch must see the GPU or YOLO silently falls back to CPU."""
    import torch

    if not torch.cuda.is_available():
        raise ColabSetupError("PyTorch cannot use the assigned CUDA GPU.")
    return torch.cuda.get_device_name(0)
