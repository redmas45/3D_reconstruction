"""Enforces the §3 process boundary for the one module Blender shares with the host.

`blender/service.py` imports `blender_protocol` by path because a wire format must have
exactly one definition — duplicating it would guarantee host and service drift apart.
That exception is only safe while the module stays stdlib-only and free of business
logic, so both properties are tested rather than trusted.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_MODULE = PROJECT_ROOT / "backend" / "infrastructure" / "blender_protocol.py"
BLENDER_SERVICE_MODULE = PROJECT_ROOT / "backend" / "legacy" / "blender" / "service.py"
WARM_SHELL_MODULE = PROJECT_ROOT / "backend" / "legacy" / "blender" / "warm_shell.py"

PERMITTED_PROTOCOL_IMPORTS = frozenset({"json", "dataclasses", "typing"})

# Anything under backend/ other than the protocol module itself is business logic.
FORBIDDEN_IMPORT_ROOTS = frozenset({
    "domain", "application", "interfaces", "numpy", "cv2", "ultralytics",
})


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


class TestProtocolModuleStaysShareable:
    def test_protocol_imports_only_the_standard_library(self):
        assert _module_imports(PROTOCOL_MODULE) <= PERMITTED_PROTOCOL_IMPORTS

    def test_protocol_imports_no_project_code(self):
        assert not _module_imports(PROTOCOL_MODULE) & FORBIDDEN_IMPORT_ROOTS

    def test_protocol_is_importable_standalone(self):
        """Blender adds only the module's directory to sys.path, not the package."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; sys.path.insert(0, r'%s'); "
                "import blender_protocol; print(blender_protocol.PROTOCOL_VERSION)"
                % PROTOCOL_MODULE.parent,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1"


class TestBlenderSideStaysIsolated:
    @pytest.mark.parametrize("module_path", [BLENDER_SERVICE_MODULE, WARM_SHELL_MODULE])
    def test_blender_modules_import_no_business_logic(self, module_path):
        assert not _module_imports(module_path) & FORBIDDEN_IMPORT_ROOTS

    def test_service_imports_only_the_protocol_from_src(self):
        imports = _module_imports(BLENDER_SERVICE_MODULE)
        assert "blender_protocol" in imports
        assert not imports & FORBIDDEN_IMPORT_ROOTS

    def test_warm_shell_does_not_import_the_protocol(self):
        # Scene construction has no business knowing about the wire format.
        assert "blender_protocol" not in _module_imports(WARM_SHELL_MODULE)
