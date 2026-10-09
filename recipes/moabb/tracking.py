"""Result files, the environment they were measured in, and optional W&B logging (offline on clusters)."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def _command(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device):
    """Host, GPU (with its MIG profile), library versions, git commit and Slurm job."""
    import braindecode
    import mne
    import moabb

    env = dict(host=platform.node(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), braindecode=braindecode.__version__,
               moabb=moabb.__version__, mne=mne.__version__,
               commit=_command("git", "rev-parse", "HEAD"),
               dirty=bool(_command("git", "status", "--porcelain", "--untracked-files=no")))
    if device.type == "cuda":
        env.update(gpu=torch.cuda.get_device_name(device), gpu_memory=torch.cuda.mem_get_info(device)[1],
                   driver=_command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                   gpu_list=_command("nvidia-smi", "-L"))  # lists the MIG profile, e.g. "MIG 2g.20gb Device 0"
    for key in ("SLURM_CLUSTER_NAME", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME"):
        if key in os.environ:
            env[key.lower()] = os.environ[key]
    return env


def write_csv(path, rows):
    """Rows as a CSV; the columns are every key of every row, in order of appearance."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_environment(out, env):
    """``<out>-env.json``, next to the results it describes."""
    path = out.with_name(f"{out.stem}-env.json")
    path.write_text(json.dumps(env, indent=2))
    return path


def start_run(project, job_type, hparams_path, hparams, args, env):
    """A W&B run, or None when ``project`` is None. Set WANDB_MODE=offline and sync later on clusters."""
    if project is None:
        return None
    import wandb

    config = dict(hparams=hparams, args={k: str(v) for k, v in vars(args).items()}, environment=env)
    return wandb.init(project=project, job_type=job_type, group=Path(hparams_path).stem, config=config)


def finish_run(run, rows, paths, summary=None):
    """Log the rows as a table and the result files as one artifact, then finish the run."""
    if run is None:
        return
    import wandb

    fields = list(dict.fromkeys(k for row in rows for k in row))
    run.summary.update(summary or {})
    run.log({"rows": wandb.Table(columns=fields, data=[[row.get(k) for k in fields] for row in rows])})
    if paths:
        artifact = wandb.Artifact(f"{run.group}-{run.id}", type="results")
        for path in paths:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    run.finish()
