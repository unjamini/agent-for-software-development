from __future__ import annotations

import subprocess
import sys
import textwrap
import tempfile
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class PythonInterpreterInput(BaseModel):
    code: str = Field(
        ...,
        description="Python code snippet to execute",
    )


@dataclass
class PythonSandbox:
    timeout_seconds: int = 10

    def __call__(self, code: str) -> str:
        formatted = textwrap.dedent(code).strip()
        if not formatted:
            return "No code supplied."

        with tempfile.TemporaryDirectory(prefix="agent_pytool_") as workdir:
            work_path = Path(workdir)
            script_path = work_path / "snippet.py"
            script_path.write_text(formatted, encoding="utf-8")

            cmd = [sys.executable, str(script_path)]

            try:
                completed = subprocess.run(
                    cmd,
                    cwd=work_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return f"Execution timed out after {self.timeout_seconds} seconds."

            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if completed.returncode != 0:
                return f"Error (exit {completed.returncode}):\n{stderr or stdout}"

            if stdout:
                return stdout
            if stderr:
                return stderr
            return "Execution completed with no output."


def create_python_tool(timeout_seconds: int = 10) -> StructuredTool:
    sandbox = PythonSandbox(timeout_seconds=timeout_seconds)
    return StructuredTool.from_function(
        name="python_interpreter",
        description=(
            "Еxe Python code"
        ),
        func=sandbox,
        args_schema=PythonInterpreterInput,
    )


