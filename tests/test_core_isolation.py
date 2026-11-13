"""The analysis core imports nothing from verdict_gate.adapters or the CLI, and no HTTP client.

Checked statically (every import statement in src/verdict_gate/core/) and at runtime (importing every core module in a
fresh interpreter loads none of those packages, nor an HTTP client).
"""

import ast
import subprocess
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "verdict_gate" / "core"
FORBIDDEN = ("verdict_gate.adapters", "verdict_gate.cli")
FORBIDDEN_RUNTIME = FORBIDDEN + ("httpx", "requests", "openai", "metaskill", "noisegate")


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import inside verdict_gate.core
                base = "verdict_gate.core" if node.level == 1 else "verdict_gate"
                names.append(f"{base}.{node.module}" if node.module else base)
            else:
                names.append(node.module or "")
    return names


def test_core_modules_exist():
    expected = {"schemas", "paired", "harm", "posterior", "calibrate", "sequential", "decision", "scope"}
    assert expected <= {p.stem for p in CORE.glob("*.py")}


def test_core_has_no_forbidden_imports():
    bad = [(p.name, name) for p in CORE.glob("*.py") for name in _imports(p)
           if any(name == f or name.startswith(f + ".") for f in FORBIDDEN)]
    assert bad == []


def test_importing_core_loads_no_forbidden_package():
    modules = [f"verdict_gate.core.{p.stem}" for p in CORE.glob("*.py") if p.stem != "__init__"]
    code = "import sys\n" + "".join(f"import {m}\n" for m in modules) + "print('\\n'.join(sys.modules))"
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    bad = [m for m in loaded if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_RUNTIME)]
    assert bad == []
