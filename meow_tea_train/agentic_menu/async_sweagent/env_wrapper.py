from pathlib import Path
from typing import Iterable, List, Mapping, Any

import os, stat, time
from shutil import rmtree


from sweagent.run.batch_instances import BatchInstance
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.environment.repo import GithubRepoConfig
from sweagent.environment.conda import CondaDeploymentConfig
from sweagent.agent.problem_statement import TextProblemStatement


def batch_instance_from_dict(
    d: Mapping[str, Any],
    *,
    deployment_cfg: CondaDeploymentConfig | None = None,
) -> BatchInstance:
    """
    Required keys in `d`:
      - instance_id, repo, base_commit, problem_statement
    Optional keys:
      - test_patch, eval_script, FAIL_TO_PASS, PASS_TO_PASS

    Assumptions:
      - repo is GitHub (string URL or {'github_url': ...})
      - CondaDeployment only
    """
    base_commit = str(d.get("base_commit", "HEAD"))
    repo = str(d.get("repo", ""))
    github_url = f"https://github.com/{repo}"
    repo_cfg = GithubRepoConfig(github_url=github_url, base_commit=base_commit)
    
    if deployment_cfg is None:
        deployment_cfg = CondaDeploymentConfig() # default python=3.11

    env_cfg = EnvironmentConfig(
        deployment=deployment_cfg.model_copy(deep=True),
        repo=repo_cfg,
    )

    ps = TextProblemStatement(
        text=str(d["problem_statement"]),
        id=str(d["instance_id"]),
        extra_fields={"base_commit": base_commit},
    )

    return BatchInstance(
        env=env_cfg,
        problem_statement=ps,
        test_patch=d.get("test_patch"),
        eval_script=d.get("eval_script"),
        FAIL_TO_PASS=[str(x) for x in d.get("FAIL_TO_PASS", [])],
        PASS_TO_PASS=[str(x) for x in d.get("PASS_TO_PASS", [])],
    )


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def remove_runtime_root(runtime_root: Path, traj_root: Path, retries: int = 3, delay: float = 0.2):
    # 1) sanity: must exist, be a dir, and live under the expected traj_root
    if not runtime_root.exists() or not runtime_root.is_dir():
        return
    if not _is_under(runtime_root, traj_root):
        raise RuntimeError(f"Refusing to delete {runtime_root}; not under {traj_root}")

    # 2) try to delete whole dir; fix read-only perms if needed
    for i in range(retries):
        try:
            rmtree(runtime_root)
            return
        except Exception:
            # relax perms and retry
            for root, dirs, files in os.walk(runtime_root, topdown=False):
                for name in files + dirs:
                    p = Path(root, name)
                    try:
                        p.chmod(stat.S_IRWXU)
                    except Exception:
                        pass
            if i == retries - 1:
                raise
            time.sleep(delay)
