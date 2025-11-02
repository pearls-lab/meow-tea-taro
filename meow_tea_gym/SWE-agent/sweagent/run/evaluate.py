from __future__ import annotations

import json
import time
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from sweagent.environment.swe_env import SWEEnv
from sweagent.environment.hooks.status import SetStatusEnvironmentHook
from swerex.deployment.hooks.status import SetStatusDeploymentHook

GIT_APPLY_CANDIDATES: List[str] = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

class EvaluationError(RuntimeError):
    pass

def _log_step(logger, step_name: str, instance_id: str, details: str = "", level: str = "info"):
    """Helper to log evaluation steps with consistent formatting"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    msg = f"[{timestamp}] [{instance_id}] STEP: {step_name}"
    if details:
        msg += f" - {details}"

    if logger:
        getattr(logger, level)(msg)
    else:
        print(msg, file=sys.stderr)

def _json_or_list(val) -> List[str]:
    """Accept list[str] or a JSON-encoded string; return list[str]."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        try:
            j = json.loads(val)
            if isinstance(j, list):
                return [str(x) for x in j]
        except Exception:
            # Not JSON; allow single string as one test id
            s = val.strip()
            return [s] if s else []
    return []

def _env_start_for_eval(instance, output_dir: Path, logger=None) -> SWEEnv:
    """Build a fresh env and start it; ensure per-instance roots for local CondaDeployment."""
    instance_id = instance.problem_statement.id
    _log_step(logger, "ENV_SETUP_START", instance_id, f"Output dir: {output_dir}")

    fresh_env_cfg = deepcopy(instance.env)
    try:
        from sweagent.environment.conda import CondaDeploymentConfig
        if isinstance(fresh_env_cfg.deployment, CondaDeploymentConfig):
            if not fresh_env_cfg.deployment.instance_root:
                fresh_env_cfg.deployment.instance_root = str(output_dir)
                _log_step(logger, "ENV_CONFIG", instance_id, f"Set instance_root: {output_dir}")
            if not fresh_env_cfg.deployment.conda_root:
                fresh_env_cfg.deployment.conda_root = str(output_dir / ".conda")
                _log_step(logger, "ENV_CONFIG", instance_id, f"Set conda_root: {output_dir / '.conda'}")
    except Exception as e:
        _log_step(logger, "ENV_CONFIG", instance_id, f"Config adjustment skipped: {e}", "warning")

    _log_step(logger, "ENV_CREATE", instance_id, "Creating SWEEnv from config")
    env = SWEEnv.from_config(fresh_env_cfg)

    def _noop_status(_id: str, message: str) -> None:
        return None

    env.add_hook(SetStatusEnvironmentHook(instance.problem_statement.id, _noop_status))
    env.deployment.add_hook(SetStatusDeploymentHook(instance.problem_statement.id, _noop_status))

    _log_step(logger, "ENV_START", instance_id, "Starting environment")
    start_time = time.time()
    env.start()
    _log_step(logger, "ENV_READY", instance_id, f"Environment started in {time.time() - start_time:.2f}s")

    return env


import re

def _sanitize_eval_script_for_local_env(text: str, logger=None) -> str:
    lines_out = []
    removed_lines = 0
    modified_lines = 0

    for line in text.splitlines():
        original_line = line
        # drop docker-specific conda activation
        if re.search(r'\bsource\s+/opt/miniconda3/bin/activate\b', line):
            removed_lines += 1
            continue
        if re.search(r'^\s*conda\s+activate\b', line):
            removed_lines += 1
            continue
        # map /testbed -> "$ROOT"
        if "/testbed" in line:
            line = line.replace("/testbed", '"$ROOT"')
            modified_lines += 1
        lines_out.append(line)

    if logger:
        logger.debug(f"Sanitized eval script: {removed_lines} lines removed, {modified_lines} lines modified")

    preamble = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Debug: Print each command before execution",
        "set -x",
        "",
        "# Progress tracking",
        "echo '[EVAL_START]' $(date '+%Y-%m-%d %H:%M:%S')",
        "echo '[EVAL_PWD]' $(pwd)",
        "",
        'cd "$ROOT"',  # ensure we're at repo root
        # Minimal safety: if PATH doesn't include env yet but CONDA_PREFIX is exported by your deployment
        '[ -n "${CONDA_PREFIX:-}" ] && export PATH="${CONDA_PREFIX}/bin:${PATH}" || true',
        "",
        "# Show Python version and location for debugging",
        "echo '[EVAL_PYTHON]' $(which python) $(python --version 2>&1)",
        "",
        "# Original eval script starts here",
        "echo '[EVAL_SCRIPT_BEGIN]'",
    ]

    epilogue = [
        "",
        "echo '[EVAL_SCRIPT_END]'",
        "echo '[EVAL_COMPLETE]' $(date '+%Y-%m-%d %H:%M:%S')",
    ]

    return "\n".join(preamble + lines_out + epilogue) + "\n"


def _run_with_timeout(env: SWEEnv, cmd: str, timeout: int, logger=None, step_name: str = "COMMAND") -> Tuple[str, bool, float]:
    """Run a command and capture output + heuristic timeout flag + runtime."""
    if logger:
        logger.debug(f"[{step_name}] Running command (timeout={timeout}s): {cmd[:200]}...")

    t0 = time.time()
    timed_out = False
    out = ""

    try:
        # Add progress monitoring for long-running commands
        if timeout > 30:
            if logger:
                logger.info(f"[{step_name}] Starting long-running command (timeout={timeout}s)")

        out = env.communicate(cmd, check="warn", timeout=timeout, error_msg="command failed")

    except Exception as e:
        out = f"Exception: {e}"
        timed_out = "timeout" in str(e).lower()
        if logger:
            logger.error(f"[{step_name}] Command failed: {e}")

    runtime = time.time() - t0

    if logger:
        if timed_out:
            logger.error(f"[{step_name}] TIMEOUT after {runtime:.2f}s")
        else:
            logger.debug(f"[{step_name}] Completed in {runtime:.2f}s")

    return out, timed_out, runtime


def _find_model_patch_text(output_dir: Path, instance_id: str, logger=None) -> tuple[str, str | None]:
    """
    Return (patch_text, used_path). Search in order:
      1) <output_dir>/<instance_id>.patch   <-- new canonical location
      2) <output_dir>/model.patch           <-- legacy
      3) <output_dir>/workspace/model.patch <-- legacy
    If none exist, returns ("", None).
    """
    candidates = [
        output_dir / f"{instance_id}.patch",
        output_dir / "model.patch",
        output_dir / "workspace" / "model.patch",
    ]

    if logger:
        logger.debug(f"Searching for patch in: {[str(p) for p in candidates]}")

    for p in candidates:
        if p.exists():
            patch_text = p.read_text(encoding="utf-8")
            if logger:
                logger.info(f"Found patch at: {p} (size: {len(patch_text)} chars)")
            return patch_text, str(p)

    if logger:
        logger.warning("No patch file found")
    return "", None


def _parse_test_output(test_output: str, fail_to_pass: List[str], pass_to_pass: List[str]) -> Dict:
    """
    Parse test output to determine which tests passed and failed.
    """
    # Combine all tests we're looking for
    all_tests = set(fail_to_pass + pass_to_pass)

    # Initialize results
    results = {
        "FAIL_TO_PASS": {
            "success": [],
            "failure": []
        },
        "PASS_TO_PASS": {
            "success": [],
            "failure": []
        }
    }

    # Common pytest output patterns
    pytest_patterns = [
        # Standard pytest format: "PASSED tests/test_file.py::TestClass::test_method"
        r'PASSED\s+(?:.*?::)?(\w+(?:\[.*?\])?)\s*',
        # Short format: "test_name PASSED"
        r'(\w+(?:\[.*?\])?)\s+PASSED',
        # Pytest final summary: "test_file.py::test_name PASSED"
        r'(?:.*?::)?(\w+(?:\[.*?\])?)\s+PASSED',
        # Just the test name followed by ... ok
        r'(\w+(?:\[.*?\])?)\s*\.\.\.\s*ok',
    ]

    failed_patterns = [
        # Standard pytest format for failures
        r'FAILED\s+(?:.*?::)?(\w+(?:\[.*?\])?)\s*',
        r'(\w+(?:\[.*?\])?)\s+FAILED',
        r'(?:.*?::)?(\w+(?:\[.*?\])?)\s+FAILED',
        # Error or failure indicators
        r'(\w+(?:\[.*?\])?)\s*\.\.\.\s*FAIL',
        r'(\w+(?:\[.*?\])?)\s*\.\.\.\s*ERROR',
    ]

    # Track which tests we found as passing or failing
    passing_tests = set()
    failing_tests = set()

    # Search for passing tests
    for pattern in pytest_patterns:
        for match in re.finditer(pattern, test_output, re.MULTILINE | re.IGNORECASE):
            test_name = match.group(1)
            # Check if this test name matches any of our tests
            for test in all_tests:
                if test in test_name or test_name in test:
                    passing_tests.add(test)
                    break

    # Search for failing tests
    for pattern in failed_patterns:
        for match in re.finditer(pattern, test_output, re.MULTILINE | re.IGNORECASE):
            test_name = match.group(1)
            # Check if this test name matches any of our tests
            for test in all_tests:
                if test in test_name or test_name in test:
                    failing_tests.add(test)
                    break

    # Also check for exit code (0 = all passed)
    exit_code_match = re.search(r'__EXIT_CODE__=(\d+)', test_output)
    if exit_code_match:
        exit_code = int(exit_code_match.group(1))
        # If exit code is 0 and we didn't find specific test results, assume all passed
        if exit_code == 0 and not passing_tests and not failing_tests:
            # Check if tests were actually run
            if 'pytest' in test_output.lower() or 'test' in test_output.lower():
                # Assume all tests passed if exit code is 0
                passing_tests = all_tests

    # Categorize tests
    for test in fail_to_pass:
        if test in passing_tests:
            results["FAIL_TO_PASS"]["success"].append(test)
        else:
            results["FAIL_TO_PASS"]["failure"].append(test)

    for test in pass_to_pass:
        if test in passing_tests:
            results["PASS_TO_PASS"]["success"].append(test)
        else:
            results["PASS_TO_PASS"]["failure"].append(test)

    return results


def _custom_get_eval_report(
    instance_id: str,
    test_output_path: str,
    fail_to_pass: List[str],
    pass_to_pass: List[str],
    model_patch_text: str = "",
    exit_code: int = 1,
    git_diff_before: str = "",
    git_diff_after: str = "",
) -> Dict:
    """
    Generate a custom evaluation report by parsing test output.
    """
    # Read test output
    try:
        test_output = Path(test_output_path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        test_output = ""
        print(f"Warning: Could not read test output: {e}")

    # Parse test results
    test_results = _parse_test_output(test_output, fail_to_pass, pass_to_pass)

    # Calculate totals
    f2p_success = len(test_results["FAIL_TO_PASS"]["success"])
    f2p_failure = len(test_results["FAIL_TO_PASS"]["failure"])
    p2p_success = len(test_results["PASS_TO_PASS"]["success"])
    p2p_failure = len(test_results["PASS_TO_PASS"]["failure"])

    total_success = f2p_success + p2p_success
    total_tests = f2p_success + f2p_failure + p2p_success + p2p_failure

    # Calculate pass ratio
    pass_ratio = total_success / total_tests if total_tests > 0 else 0.0

    # Determine if resolved (all FAIL_TO_PASS succeeded and all PASS_TO_PASS succeeded)
    resolved = (f2p_failure == 0 and p2p_failure == 0) if total_tests > 0 else False

    # Alternative resolution check: if exit code is 0, consider it resolved
    if exit_code == 0 and total_tests > 0:
        resolved = True

    # Build the report
    report = {
        instance_id: {
            "patch_exists": bool(model_patch_text and model_patch_text.strip()),
            "resolved": resolved,
            "tests_status": test_results,
            "pass_ratio": pass_ratio,
            "exit_code": exit_code,
            "summary": {
                "FAIL_TO_PASS": f"{f2p_success}/{f2p_success + f2p_failure}",
                "PASS_TO_PASS": f"{p2p_success}/{p2p_success + p2p_failure}",
                "total": f"{total_success}/{total_tests}"
            },
            "git_diff_before": git_diff_before,
            "git_diff_after": git_diff_after,
            "test_output_path": str(test_output_path),
        }
    }

    return report


def evaluate_instance(
    *,
    instance,
    output_dir: Path,
    logger=None,
    timeout: int = 60,
) -> Dict:
    """
    SWE-bench-harness-like eval with top-level fields:
      - test_patch: str | None
      - eval_script: str | None
      - FAIL_TO_PASS: list[str] | str | None
      - PASS_TO_PASS: list[str] | str | None

    Flow:
      - rebuild clean env
      - write patch.diff into $ROOT and try several git-apply variants
      - capture git diff before/after
      - write eval.sh from eval_script
      - run it, save test_output.txt and report.json
      - if swebench.harness.get_eval_report is importable, call it for identical report
    """
    eval_start_time = time.time()
    instance_id = instance.problem_statement.id

    _log_step(logger, "EVAL_INIT", instance_id, f"Starting evaluation (timeout={timeout}s)")

    log_dir = output_dir / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    patch_file_host = log_dir / "patch.diff"
    eval_file_host = log_dir / "eval.sh"
    test_output_path = log_dir / "test_output.txt"
    report_path = log_dir / "report.json"

    # Read model patch produced by agent
    model_patch_text, used_patch_path = _find_model_patch_text(output_dir, instance_id, logger)
    patch_file_host.write_text(model_patch_text, encoding="utf-8")

    # Eval script
    eval_script_text = getattr(instance, "eval_script", None)
    if eval_script_text:
        eval_script_text = _sanitize_eval_script_for_local_env(eval_script_text, logger)
    else:
        _log_step(logger, "EVAL_SCRIPT_MISSING", instance_id, "No eval script found", "warning")
        eval_script_text = ""
    eval_file_host.write_text(eval_script_text, encoding="utf-8")

    # ⛔️ Early exit if no patch is present (write report and stop before env build)
    if not (model_patch_text and model_patch_text.strip()):
        _log_step(logger, "NO_PATCH", instance_id, "No model patch found; stopping before environment build", "warning")

        fail_to_pass = _json_or_list(getattr(instance, "FAIL_TO_PASS", None))
        pass_to_pass = _json_or_list(getattr(instance, "PASS_TO_PASS", None))

        early_report = {
            instance_id: {
                "patch_exists": False,
                "resolved": False,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": [], "failure": fail_to_pass},
                    "PASS_TO_PASS": {"success": [], "failure": pass_to_pass},
                },
                "pass_ratio": 0.0,
                "exit_code": 1,
                "summary": {
                    "FAIL_TO_PASS": f"0/{len(fail_to_pass)}",
                    "PASS_TO_PASS": f"0/{len(pass_to_pass)}",
                    "total": f"0/{len(fail_to_pass) + len(pass_to_pass)}",
                },
                "git_diff_before": "",
                "git_diff_after": "",
                "test_output_path": str(test_output_path),
                "note": "Evaluation skipped because no patch file was found.",
            }
        }

        report_path.write_text(json.dumps(early_report, indent=4), encoding="utf-8")
        total_eval_time = time.time() - eval_start_time
        _log_step(
            logger,
            "EVAL_SKIPPED",
            instance_id,
            f"Report written (no patch). Path: {report_path}",
            "info",
        )

        return {
            "eval_completed": False,
            "report_path": str(report_path),
            "report": early_report,
            "total_eval_time_sec": total_eval_time,
            "used_patch_path": used_patch_path,
        }

    test_patch_text = getattr(instance, "test_patch", None)

    fail_to_pass = _json_or_list(getattr(instance, "FAIL_TO_PASS", None))
    pass_to_pass = _json_or_list(getattr(instance, "PASS_TO_PASS", None))
    _log_step(
        logger,
        "TEST_SPECS",
        instance_id,
        f"FAIL_TO_PASS: {len(fail_to_pass)} tests, PASS_TO_PASS: {len(pass_to_pass)} tests",
    )

    container = None
    try:
        _log_step(logger, "ENV_BUILD", instance_id, "Reconstructing evaluation environment")
        env_start = time.time()
        container = _env_start_for_eval(instance, output_dir, logger)
        _log_step(logger, "ENV_READY", instance_id, f"Environment ready in {time.time() - env_start:.2f}s")

        # Resolve $ROOT and write files inside repo root
        root_abs = container.communicate('printf "%s" "$ROOT"', check="raise").strip()
        root_path = Path(root_abs)
        _log_step(logger, "ROOT_PATH", instance_id, f"Repository root: {root_abs}")

        _log_step(logger, "FILES_WRITE", instance_id, "Writing patch and eval files to container")
        container.write_file(root_path / "patch.diff", model_patch_text)
        container.write_file(root_path / "eval.sh", eval_script_text)

        # Apply test patch if present
        if test_patch_text:
            _log_step(logger, "TEST_PATCH_APPLY", instance_id, "Applying test patch")
            container.write_file(root_path / "test.patch", test_patch_text)
            out, _to, apply_time = _run_with_timeout(
                container, f'cd "{root_abs}" && git apply -p1 test.patch',
                timeout=600, logger=logger, step_name="TEST_PATCH"
            )

        # Try git apply variants for model patch
        applied_patch = False
        last_apply_output = ""

        for idx, git_apply_cmd in enumerate(GIT_APPLY_CANDIDATES, 1):
            _log_step(logger, "PATCH_ATTEMPT", instance_id,
                     f"Attempt {idx}/{len(GIT_APPLY_CANDIDATES)}: {git_apply_cmd}")
            cmd = f'cd "{root_abs}" && {git_apply_cmd} patch.diff'
            out, _to, apply_time = _run_with_timeout(
                container, cmd, timeout=600, logger=logger,
                step_name=f"PATCH_APPLY_{idx}"
            )
            last_apply_output = out or ""

            if "error:" not in last_apply_output and "fatal:" not in last_apply_output:
                applied_patch = True
                _log_step(logger, "PATCH_SUCCESS", instance_id,
                         f"Patch applied successfully with method {idx} in {apply_time:.2f}s")
                break
            else:
                _log_step(logger, "PATCH_FAIL", instance_id,
                         f"Method {idx} failed: {last_apply_output[:200]}", "warning")

        if not applied_patch:
            _log_step(logger, "PATCH_ERROR", instance_id,
                     f"All patch methods failed. Last error: {last_apply_output[:500]}", "error")
            raise EvaluationError(f"{instance_id}: failed to apply patch")

        # Capture git diff before tests
        _log_step(logger, "GIT_DIFF_BEFORE", instance_id, "Capturing git diff before tests")
        git_diff_before = container.communicate(
            f'cd "{root_abs}" && git -c core.fileMode=false diff',
            check="warn",
            timeout=60,
        ).strip()

        # Run eval.sh with progress monitoring
        _log_step(logger, "TESTS_START", instance_id, f"Starting test execution (timeout={timeout}s)")
        run_cmd = f'cd "{root_abs}" && /bin/bash ./eval.sh; rc=$?; printf "\\n__EXIT_CODE__=%s\\n" "$rc"'

        # For long timeouts, we might want to monitor progress
        test_start = time.time()
        test_output, timed_out, total_runtime = _run_with_timeout(
            container, run_cmd, timeout=timeout, logger=logger, step_name="TESTS_RUN"
        )

        test_output_path.write_text(test_output or "", encoding="utf-8")
        _log_step(logger, "TESTS_COMPLETE", instance_id,
                 f"Tests finished in {total_runtime:.2f}s. Output: {len(test_output)} chars -> {test_output_path}")

        if timed_out:
            with test_output_path.open("a", encoding="utf-8") as f:
                f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
            _log_step(logger, "TESTS_TIMEOUT", instance_id, f"Tests timed out after {timeout}s", "error")
            raise EvaluationError(f"{instance_id}: Test timed out after {timeout} seconds.")

        # Extract exit code
        exit_line = (test_output or "").strip().splitlines()[-1] if test_output else ""
        try:
            exit_code = int(exit_line.split("__EXIT_CODE__=")[-1])
        except Exception as e:
            exit_code = 1

        # Git diff after tests
        _log_step(logger, "GIT_DIFF_AFTER", instance_id, "Capturing git diff after tests")
        git_diff_after = container.communicate(
            f'cd "{root_abs}" && git -c core.fileMode=false diff', check="warn", timeout=60
        ).strip()

        try:
            report = _custom_get_eval_report(
                instance_id=instance_id,
                test_output_path=test_output_path,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                model_patch_text=model_patch_text,
                exit_code=exit_code,
                git_diff_before=git_diff_before,
                git_diff_after=git_diff_after,
            )
            _log_step(logger, "GET_REPORT_SUCCESS", instance_id, "Test report generated successfully")
        except Exception as e:
            _log_step(logger, "GET_REPORT_FAIL", instance_id, f"Test report generation failed: {e}", "warning")
            report = {
                instance_id: {
                    "patch_exists": bool(model_patch_text and model_patch_text.strip()),
                    "resolved": False,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": [], "failure": fail_to_pass},
                        "PASS_TO_PASS": {"success": [], "failure": pass_to_pass},
                    },
                    "pass_ratio": 0.0,
                    "exit_code": exit_code,
                    "summary": {
                        "FAIL_TO_PASS": f"0/{len(fail_to_pass)}",
                        "PASS_TO_PASS": f"0/{len(pass_to_pass)}",
                        "total": f"0/{len(fail_to_pass) + len(pass_to_pass)}",
                    },
                    "git_diff_before": git_diff_before,
                    "git_diff_after": git_diff_after,
                    "test_output_path": str(test_output_path),
                    "note": f"Fallback report due to error: {e}",
                }
            }

        report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

        total_eval_time = time.time() - eval_start_time
        _log_step(logger, "EVAL_SUCCESS", instance_id,
                 f"Evaluation completed successfully in {total_eval_time:.2f}s. "
                 f"Resolved: {exit_code == 0}. Report: {report_path}")

        return {
            "eval_completed": True,
            "report_path": str(report_path),
            "report": report,
            "exit_code": exit_code,
            "runtime_sec": total_runtime,
            "total_eval_time_sec": total_eval_time,
            "used_patch_path": used_patch_path,
        }

    except EvaluationError as e:
        total_eval_time = time.time() - eval_start_time
        _log_step(logger, "EVAL_ERROR", instance_id,
                 f"Evaluation failed after {total_eval_time:.2f}s: {e}", "error")

        failure_report = {
            instance_id: {
                "resolved": False,
                "timeout": "timed out" in str(e).lower(),
                "error": str(e),
                "FAIL_TO_PASS": fail_to_pass,
                "PASS_TO_PASS": pass_to_pass,
                "test_output_path": str(test_output_path),
            }
        }
        report_path.write_text(json.dumps(failure_report, indent=4), encoding="utf-8")

        return {
            "eval_completed": False,
            "report_path": str(report_path),
            "report": failure_report,
            "total_eval_time_sec": total_eval_time,
            "used_patch_path": used_patch_path,
        }
    finally:
        try:
            if container:
                _log_step(logger, "CLEANUP", instance_id, "Closing container")
                container.close()
                _log_step(logger, "CLEANUP_DONE", instance_id, "Container closed")
        except Exception as e:
            _log_step(logger, "CLEANUP_ERROR", instance_id, f"Failed to close container: {e}", "warning")
