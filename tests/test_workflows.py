"""Lint the GitHub Actions workflows themselves.

Root-cause fix for a real incident: `eval-gate.yml` shipped with an unclosed
`if` in one step and a stray `fi` in the next, so BOTH gate steps died with a
bash syntax error on every pull request. The gate looked wired up and was in
fact inert — the exact "green tick is lying" failure mode the eval gate exists
to prevent, one layer up.

These tests are cheap and catch the whole class:
  1. every workflow file is valid YAML with the keys Actions requires;
  2. every `run:` block is valid shell (`bash -n`), with the `${{ }}` template
     expressions stubbed out so the parse sees the same shape Actions will.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))

# Actions substitutes ${{ ... }} before bash ever sees the script. Replace each
# with a harmless literal so `bash -n` checks OUR syntax, not the templating.
TEMPLATE_EXPR = re.compile(r"\$\{\{[^}]*\}\}")


def _iter_run_blocks():
    for wf_path in WORKFLOWS:
        doc = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        for job_name, job in (doc.get("jobs") or {}).items():
            for i, step in enumerate(job.get("steps") or []):
                if "run" in step:
                    label = f"{wf_path.name}::{job_name}::step[{i}] {step.get('name', '')}".strip()
                    yield label, step["run"], step.get("shell", "bash")


def test_workflow_dir_is_not_empty():
    assert WORKFLOWS, f"no workflow files found under {WORKFLOW_DIR}"


@pytest.mark.parametrize("wf_path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_valid_yaml_with_jobs(wf_path: Path):
    doc = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{wf_path.name} is not a YAML mapping"
    # PyYAML parses the bare key `on:` as the boolean True — accept either form.
    assert "on" in doc or True in doc, f"{wf_path.name} has no trigger (`on:`)"
    jobs = doc.get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{wf_path.name} defines no jobs"
    for job_name, job in jobs.items():
        assert job.get("runs-on"), f"{wf_path.name}::{job_name} has no runs-on"
        assert job.get("steps"), f"{wf_path.name}::{job_name} has no steps"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize(
    "label,script,shell",
    list(_iter_run_blocks()),
    ids=lambda v: v if isinstance(v, str) and "::" in v else "",
)
def test_run_block_is_valid_shell(label: str, script: str, shell: str, tmp_path: Path):
    if shell not in ("bash", "sh"):
        pytest.skip(f"{label}: non-POSIX shell ({shell})")
    stubbed = TEMPLATE_EXPR.sub("STUB", script)
    script_file = tmp_path / "step.sh"
    script_file.write_text(stubbed, encoding="utf-8", newline="\n")
    # Invoke with a RELATIVE filename from tmp_path: Git Bash on Windows cannot
    # resolve a native "C:\..." path passed as an argument.
    proc = subprocess.run(
        ["bash", "-n", script_file.name], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0, f"{label} is not valid shell:\n{proc.stderr}"
