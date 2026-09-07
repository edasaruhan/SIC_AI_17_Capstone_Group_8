"""Exercise the long-job runner without a GPU, downloads, or real training."""

import fcntl
import os
import runpy
import sys

import pytest

from evidence_eval.io import read_json, write_json

JOB = runpy.run_path("scripts/run_english_listwise.py")


def test_fixed_english_config():
    config = JOB["config_for"]("fixture")
    assert config["experiment"] == "fixture"
    assert (config["track"], config["category"], config["seed"], config["epochs"]) == (
        "en",
        "vpn",
        7,
        2,
    )
    assert len(config["revision"]) == 40
    assert config["smoke"] is False


def test_progress_reads_completed_and_running_folds(tmp_path):
    write_json(tmp_path / "named_fold0/state.json", {"status": "completed"})
    write_json(tmp_path / "masked_fold0/state.json", {"status": "running"})
    assert JOB["progress"](tmp_path) == {
        "completed_folds": ["named_fold0"],
        "running_folds": ["masked_fold0"],
    }


def test_duplicate_job_cannot_overwrite_status(tmp_path):
    job = tmp_path / "training_jobs" / JOB["JOB_NAME"]
    write_json(job / "status.json", {"status": "running", "sentinel": "preserved"})
    with (job / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            JOB["run"](tmp_path)
        assert JOB["job_status"](tmp_path)["job_active"] is True
    assert read_json(job / "status.json")["sentinel"] == "preserved"
    assert JOB["job_status"](tmp_path)["status"] == "interrupted"


@pytest.mark.parametrize("return_code", [0, 3])
def test_stage_records_output_and_propagates_exit_code(tmp_path, return_code):
    state = {"status": "running", "run_dir": str(tmp_path / "run")}
    command = [sys.executable, "-c", f"print('fixture child'); raise SystemExit({return_code})"]
    with (tmp_path / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = ("fixture", command, os.environ.copy(), tmp_path, state, lock.fileno())
        if return_code:
            with pytest.raises(RuntimeError, match="exit code 3"):
                JOB["execute_stage"](*args, poll_seconds=0.01)
        else:
            JOB["execute_stage"](*args, poll_seconds=0.01)
    assert "fixture child" in (tmp_path / "fixture.log").read_text()
    assert state["worker_pid"] is None
    assert read_json(tmp_path / "status.json")["stage"] == "fixture"
