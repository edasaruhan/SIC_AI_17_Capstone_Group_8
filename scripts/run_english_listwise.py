"""Offline EN VPN seed-7 job: smoke, train, audit, with a lock and live status.

Run from the repository root. Download the pinned encoder separately first.
This runner never sources .env, downloads models, or calls collection APIs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from evidence_eval.io import digest, read_json, write_json
from evidence_eval.workspace import DEFAULT_ROOT, verify

REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
JOB_NAME = "en_vpn_seed7"


def now() -> str:
    return datetime.now(UTC).isoformat()


def config_for(identity: str) -> dict:
    return {
        "experiment": identity,
        "track": "en",
        "category": "vpn",
        "encoder": "bert-base-uncased",
        "revision": REVISION,
        "seed": 7,
        "epochs": 2,
        "max_length": 128,
        "candidate_batch": 4,
        "smoke": False,
        "learning_rate": 2e-5,
        "objective": "response_listwise_softmax",
    }


def progress(run: Path) -> dict:
    completed, active = [], []
    for path in sorted(run.glob("*_fold*/state.json")):
        value = read_json(path)
        if value["status"] == "completed":
            completed.append(path.parent.name)
        elif value["status"] == "running":
            active.append(path.parent.name)
    return {"completed_folds": completed, "running_folds": active}


def job_status(root: Path) -> dict:
    job = root / "training_jobs" / JOB_NAME
    value = read_json(job / "status.json")
    with (job / "job.lock").open() as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            value["job_active"] = True
        else:
            value["job_active"] = False
            fcntl.flock(lock, fcntl.LOCK_UN)
            if value["status"] == "running":
                value["status"] = "interrupted"
    return value


def execute_stage(
    name: str,
    command: list[str],
    env: dict[str, str],
    job: Path,
    state: dict,
    lock_fd: int,
    *,
    poll_seconds: float = 10,
) -> None:
    state.update(stage=name, worker_pid=None, heartbeat_at=now())
    write_json(job / "status.json", state)
    logger.info("Starting {}. Output: {}", name, job / f"{name}.log")
    started = time.monotonic()
    last_notice = started
    with (job / f"{name}.log").open("a", encoding="utf-8") as output:
        output.write(f"\n--- {now()} stage={name} ---\n")
        output.flush()
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            pass_fds=(lock_fd,),
        )
        try:
            state["worker_pid"] = process.pid
            while process.poll() is None:
                state.update(
                    heartbeat_at=now(),
                    stage_elapsed_seconds=round(time.monotonic() - started),
                    progress=progress(Path(state["run_dir"])),
                )
                write_json(job / "status.json", state)
                if time.monotonic() - last_notice >= 60:
                    logger.info(
                        "{} alive; elapsed={}s; completed_folds={}; active={}",
                        name,
                        state["stage_elapsed_seconds"],
                        len(state["progress"]["completed_folds"]),
                        state["progress"]["running_folds"],
                    )
                    last_notice = time.monotonic()
                time.sleep(poll_seconds)
            if process.returncode != 0:
                raise RuntimeError(
                    f"{name} failed with exit code {process.returncode}; see {name}.log"
                )
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            state["worker_pid"] = None
    logger.info("{} completed", name)


def run(root: Path) -> dict:
    root = root.resolve()
    job = root / "training_jobs" / JOB_NAME
    job.mkdir(parents=True, exist_ok=True)
    with (job / "job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("English training job is already active; use --status") from exc
        state = {
            "status": "running",
            "stage": "preparing",
            "started_at": now(),
            "heartbeat_at": now(),
            "pid": os.getpid(),
            "collection_api_calls": False,
        }
        write_json(job / "status.json", state)
        try:
            manifest = verify(root)
            config = config_for(manifest["identity"])
            run_dir = root / "listwise" / digest(config)[:16]
            state.update(config=config, run_dir=str(run_dir))
            write_json(job / "status.json", state)
            env = os.environ.copy()
            env.update(
                PYTHONPATH=str(Path("src").resolve()),
                HF_HOME=str(root / "hf_cache"),
                HF_HUB_OFFLINE="1",
                HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                HF_HUB_DISABLE_PROGRESS_BARS="1",
                TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="1",
            )
            base = [sys.executable, "-m", "evidence_eval", "--root", str(root)]
            options = [
                "--track",
                "en",
                "--category",
                "vpn",
                "--model-revision",
                REVISION,
                "--seed",
                "7",
                "--epochs",
                "2",
                "--max-length",
                "128",
                "--candidate-batch",
                "4",
            ]
            stages = [
                ("smoke", [*base, "m3-smoke", *options]),
                ("training", [*base, "m3-train", *options]),
                (
                    "audit",
                    [
                        sys.executable,
                        "scripts/audit_listwise_run.py",
                        "--run",
                        str(run_dir),
                        "--reload",
                    ],
                ),
            ]
            for name, command in stages:
                execute_stage(name, command, env, job, state, lock.fileno())
            audit = read_json(run_dir / "audit.json")
            if (
                audit["config"] != config
                or audit["checkpoint_reload_scope"] != "one_response_per_fold"
            ):
                raise ValueError("Final checkpoint audit does not match the training job")
            state.update(status="completed", stage="completed", finished_at=now())
        except BaseException as exc:
            state.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                error_type=type(exc).__name__,
                error=str(exc),
                finished_at=now(),
            )
            raise
        finally:
            state["heartbeat_at"] = now()
            if "run_dir" in state:
                state["progress"] = progress(Path(state["run_dir"]))
            write_json(job / "status.json", state)
        return state


def interrupted(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt("Training job interrupted; existing checkpoints preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = job_status(args.root) if args.status else run(args.root)
    except (ValueError, FileNotFoundError, RuntimeError, KeyboardInterrupt) as exc:
        logger.error("{}: {}", type(exc).__name__, exc)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
