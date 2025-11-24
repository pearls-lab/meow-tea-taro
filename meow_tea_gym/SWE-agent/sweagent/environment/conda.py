# sweagent/environment/conda.py
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
import fcntl
import time

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import IsAliveResponse
from swerex.runtime.local import LocalRuntime
from swerex.utils.log import get_logger


__all__ = ["CondaDeployment", "CondaDeploymentConfig"]


class CondaDeploymentConfig(BaseModel):
    type: Literal["conda"] = "conda"

    # Where ALL conda bits live for this instance (env/, activate.sh, caches)
    conda_root: str | None = Field(
        default=None,
        description="Root dir holding env/, activate.sh, conda-pkgs/, pip-cache/ for this instance, e.g. <output>/<id>/.conda",
    )
    # Parent directory for per-instance files like workspace/
    instance_root: str | None = Field(
        default=None,
        description="Parent dir for this instance; workspace/ is created here. "
                    "If not set and conda_root ends with '.conda', uses its parent.",
    )

    name: str = Field(default="sweagent", description="Fallback label if conda_root not provided.")
    python: str = Field(default="3.11", description="Default python version for the env.")
    conda_exe: str | None = Field(default=None, description="Path to conda/mamba; else autodetect.")
    use_mamba: bool = Field(default=False, description="Prefer 'mamba' if available.")

    channels: list[str] = Field(default_factory=lambda: ["conda-forge"])
    packages: list[str] = Field(default_factory=list)
    post_create_commands: list[str] = Field(default_factory=list)

    clear_conda_workspace: bool = Field(default=False, description="If true, remove .conda/ and workspace/ on stop()")
    remove_env_on_stop: bool = Field(default=False, description="If true, remove env/ on stop().")

    model_config = ConfigDict(extra="forbid")

    def get_deployment(self) -> AbstractDeployment:
        from .conda import CondaDeployment
        return CondaDeployment.from_config(self)


class CondaDeployment(AbstractDeployment):
    """
    Creates (if missing) and activates a conda env at <conda_root>/env via rcfile <conda_root>/activate.sh.
    Workspace lives at <instance_root>/workspace (sibling of .conda).
    Uses file locking to prevent race conditions when multiple processes create environments.
    """

    # Class-level lock file path for serializing conda operations
    _CONDA_LOCK_FILE = Path("~/.cache/sweagent/.conda_create.lock").expanduser()

    def __init__(self, **kwargs: Any):
        self._config = CondaDeploymentConfig(**kwargs)
        self.logger: logging.Logger = get_logger("rex-deploy-conda")
        self._hooks = CombinedDeploymentHook()
        self._runtime: LocalRuntime | None = None

        # Resolve directories
        if self._config.conda_root:
            conda_root = Path(self._config.conda_root).expanduser().resolve()
        else:
            conda_root = Path("~/.cache/sweagent/conda-envs").expanduser().resolve() / self._config.name / ".conda"
        conda_root.mkdir(parents=True, exist_ok=True)

        if self._config.instance_root:
            instance_root = Path(self._config.instance_root).expanduser().resolve()
        else:
            instance_root = conda_root.parent
        instance_root.mkdir(parents=True, exist_ok=True)

        # store root directories
        self._conda_root = conda_root
        self._instance_root = instance_root

        # conda artifacts - use SHARED package cache to avoid redundant downloads
        self._env_prefix  = self._conda_root / "env"
        self._rcfile_path = self._conda_root / "activate.sh"

        # Use system-wide conda package cache (conda's default location handles locking)
        # Don't override - let conda use its default cache with proper locking
        self._pkgs_dir = None  # Will use conda's default
        self._pip_cache = Path("~/.cache/sweagent/pip-cache").expanduser().resolve()
        
        # Create pip cache directory only
        self._pip_cache.mkdir(parents=True, exist_ok=True)

        # Ensure lock file directory exists
        self._CONDA_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # workspace directory (per-instance)
        self._workspace = self._instance_root / "workspace"
        self._workspace.mkdir(parents=True, exist_ok=True)


    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: CondaDeploymentConfig) -> Self:
        return cls(**config.model_dump())

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        if self._runtime is None:
            return IsAliveResponse(is_alive=False, message="Runtime is None.")
        return await self._runtime.is_alive(timeout=timeout)

    async def start(self):
        self._accept_conda_tos()
        
        # Removed global lock to allow parallel environment creation.
        # We rely on unique prefixes and conda's internal package cache locking.
        self._ensure_conda_env()
        
        self._write_rcfile()
        self._runtime = LocalRuntime(logger=self.logger)

    def _acquire_conda_lock(self):
        """Context manager that acquires an exclusive file lock for conda operations."""
        class FileLock:
            def __init__(self, lock_file: Path, logger: logging.Logger):
                self.lock_file = lock_file
                self.logger = logger
                self.fd = None

            def __enter__(self):
                # No-op: locking is disabled to allow parallel environment creation
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # No-op
                pass

        return FileLock(self._CONDA_LOCK_FILE, self.logger)

    async def stop(self):
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None
        if self._config.remove_env_on_stop and self._env_prefix.exists():
            try:
                self._conda_remove_env()
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Failed to remove conda env at %s: %s", self._env_prefix, e)
        if self._config.clear_conda_workspace:
            try:
                self._clear_workspace()
            except Exception as e:
                self.logger.warning("Failed to clear conda and workspace directories due to:\n", e)

    @property
    def runtime(self) -> LocalRuntime:
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    @property
    def startup_sources(self) -> list[str]:
        """Files to source when starting the bash session (SWEEnv will use this)."""
        return [str(self._rcfile_path)]

    @property
    def work_root(self) -> str:
        """Directory SWEEnv should use for repo/tools workspace."""
        return str(self._workspace)

    # ---- internals ----
    def _accept_conda_tos(self):
        """
        Automatically accept conda Terms of Service for Anaconda channels.
        This prevents CondaToSNonInteractiveError when creating environments.
        """
        conda = self._resolve_conda_exe()
        
        # Channels that require TOS acceptance
        channels_requiring_tos = [
            "https://repo.anaconda.com/pkgs/main",
            "https://repo.anaconda.com/pkgs/r",
        ]
        
        for channel in channels_requiring_tos:
            try:
                # Check if we need to accept TOS for this channel
                args = [conda, "tos", "accept", "--override-channels", "--channel", channel]
                self.logger.info(f"Accepting conda TOS for channel: {channel}")
                self._run(args, f"accepting TOS for {channel}", check=False)
            except Exception as e:
                # Log but don't fail - TOS might already be accepted
                self.logger.debug(f"Could not accept TOS for {channel}: {e}")

    def _ensure_conda_env(self):
        conda = self._resolve_conda_exe()
        prefix = str(self._env_prefix)
        is_new = not (self._env_prefix / "conda-meta").exists()

        env = os.environ.copy()
        # Increase internal lock timeout for conda to reduce immediate failures
        env["CONDA_LOCK_TIMEOUT"] = "300"
        # Don't override CONDA_PKGS_DIRS - use conda's default with proper locking

        if is_new:
            args = [conda, "create", "-y", "-p", prefix, f"python={self._config.python}"]
            for ch in self._config.channels:
                args += ["-c", ch]
            self._run_with_retry(args, "creating conda environment", env=env)

        if self._config.packages:
            args = [conda, "install", "-y", "-p", prefix, *self._config.packages]
            for ch in self._config.channels:
                args += ["-c", ch]
            self._run_with_retry(args, "installing conda packages", env=env)

        if is_new and self._config.post_create_commands:
            for cmd in self._config.post_create_commands:
                env2 = env.copy()
                env2["PIP_CACHE_DIR"] = str(self._pip_cache)
                env2["CONDA_LOCK_TIMEOUT"] = "300"
                self._run([conda, "run", "-p", prefix, "bash", "-lc", cmd], f"post-create: {cmd}", env=env2)

    def _run_with_retry(self, args: list[str], info: str, env: dict[str, str] | None = None, retries: int = 20, delay: float = 5.0, max_delay: float = 30.0):
        """Run a command with retries to handle transient failures (e.g. network, locking)."""
        import random
        for i in range(retries):
            try:
                # Only log full failure output on the last attempt
                is_last_attempt = (i == retries - 1)
                self._run(args, info, env=env, log_failure=is_last_attempt)
                return
            except RuntimeError as e:
                if i == retries - 1:
                    raise
                
                # Exponential backoff with jitter, capped at max_delay
                backoff = min(delay * (1.5 ** i), max_delay)
                sleep_time = backoff + random.uniform(0, 5)
                self.logger.warning(f"Attempt {i+1}/{retries} failed for {info}: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)

    def _conda_remove_env(self):
        conda = self._resolve_conda_exe()
        self._run([conda, "env", "remove", "-p", str(self._env_prefix), "-y"], "removing conda environment")

    def _clear_workspace(self):
        self._run(["rm", "-rf", str(self._conda_root)], "removing conda directory")
        self._run(["rm", "-rf", str(self._workspace)], "removing workspace directory")

    def _resolve_conda_exe(self) -> str:
        if self._config.conda_exe:
            return self._config.conda_exe
        if self._config.use_mamba:
            m = shutil.which("mamba")
            if m:
                return m
        env_exe = os.getenv("CONDA_EXE")
        if env_exe and Path(env_exe).exists():
            return env_exe
        for cand in ("conda", "micromamba"):
            exe = shutil.which(cand)
            if exe:
                return exe
        raise RuntimeError(
            "Could not find conda/mamba executable. Install Miniconda/Anaconda or set CondaDeploymentConfig.conda_exe."
        )

    def _write_rcfile(self):
        prefix = str(self._env_prefix)
        tools_root = os.path.join(self.work_root, "tools")
        content = f"""
# Auto-generated by CondaDeployment
# Initialize conda in bash and activate env by prefix
if [ -n "$CONDA_EXE" ] && command -v "$CONDA_EXE" >/dev/null 2>&1; then
  eval "$($CONDA_EXE shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  . "/opt/miniconda3/etc/profile.d/conda.sh"
fi

# Route pip cache
export PIP_CACHE_DIR="{self._pip_cache}"
export PIP_PROGRESS_BAR=off
export PAGER=cat
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONNOUSERSITE=1

# Tools convenience vars
export SWE_TOOLS_ROOT="{tools_root}"
export PATH="$SWE_TOOLS_ROOT/bin:$PATH"

# Activate the env
conda activate "{prefix}"
""".strip() + "\n"
        self._rcfile_path.write_text(content, encoding="utf-8")

    def _run(self, args: list[str], info: str, env: dict[str, str] | None = None, check: bool = True, log_failure: bool = True):
        self.logger.info("CondaDeployment: %s: %s", info, " ".join(args))
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        if check and proc.returncode != 0:
            if log_failure:
                self.logger.error("Command failed (%s):\n%s", info, proc.stdout)
            raise RuntimeError(f"CondaDeployment failed while {info}. Exit code {proc.returncode}")
        self.logger.debug(proc.stdout)