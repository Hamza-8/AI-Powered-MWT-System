"""Atomic, complete training snapshots. Only load checkpoints from trusted sources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 2


def identity_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_checkpoint(path, model, optimizers=None, schedulers=None, scaler=None, *,
                    epoch=0, step_in_epoch=0, global_step=0, metadata=None, best=float("inf"),
                    bad_epochs=0, extra=None) -> dict:
    """Position is the *next* batch to execute, not the last completed batch."""
    state = {"schema_version": SCHEMA_VERSION, "model": model.state_dict(),
             "optimizers": {k: v.state_dict() for k, v in (optimizers or {}).items()},
             "schedulers": {k: v.state_dict() for k, v in (schedulers or {}).items()},
             "scaler": scaler.state_dict() if scaler is not None else None,
             "rng": capture_rng(), "epoch": int(epoch), "step_in_epoch": int(step_in_epoch),
             "global_step": int(global_step), "metadata": metadata or {}, "best": float(best),
             "bad_epochs": int(bad_epochs), "extra": extra or {}}
    atomic_torch_save(state, path)
    return state


def load_checkpoint(path, model=None, optimizers=None, schedulers=None, scaler=None, *,
                    expected_identity=None, restore_random=True, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Checkpoint schema is not v2; legacy branch checkpoints require explicit retraining/migration")
    if expected_identity is not None and state["metadata"].get("run_identity") != expected_identity:
        raise ValueError("Checkpoint identity mismatch: data/config/splits/frequencies/preprocessing changed")
    if model is not None:
        model.load_state_dict(state["model"], strict=True)
    for key, optimizer in (optimizers or {}).items():
        optimizer.load_state_dict(state["optimizers"][key])
        # PyTorch's scheduler ordering marker is not part of optimizer.state_dict.
        # A restored completed optimizer update must retain that ordering evidence.
        if state.get("global_step", 0) > 0:
            optimizer._opt_called = True
        if state.get("global_step",0) > 0:
            optimizer._opt_called = True  # scheduler ordering instrumentation is not serialized by PyTorch
    for key, scheduler in (schedulers or {}).items():
        scheduler.load_state_dict(state["schedulers"][key])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_random:
        restore_rng(state["rng"])
    return state
