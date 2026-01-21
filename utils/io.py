"""I/O helpers for decluttering project."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, Path]


def load_array(path: PathLike) -> np.ndarray:
    """Load a numpy or HDF5 tensor from disk."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise ImportError("h5py is required to load .h5 files") from exc
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise KeyError("Expected dataset named 'data' in HDF5 file")
            return np.asarray(handle["data"])
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def save_array(path: PathLike, array: np.ndarray) -> None:
    """Save a numpy array to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, array)
        return
    if path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise ImportError("h5py is required to save .h5 files") from exc
        with h5py.File(path, "w") as handle:
            handle.create_dataset("data", data=array)
        return
    raise ValueError(f"Unsupported file extension: {path.suffix}")
