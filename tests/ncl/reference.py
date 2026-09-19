# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and orient runtime-generated NCL reference outputs for pytest."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

import spharmgrid as sg

GridKind: TypeAlias = Literal["gl", "cc"]
InputFamily: TypeAlias = Literal["analytic", "random"]
LatitudeOrder: TypeAlias = Literal["ascending", "descending"]

SCHEMA_VERSION = 1
SEED = 20260919
DOMAINS = ("T42", "T5_42", "T42x10", "R21")
GRID_KINDS: tuple[GridKind, GridKind] = ("gl", "cc")
INPUT_FAMILIES: tuple[InputFamily, InputFamily] = ("analytic", "random")


def reference_directory() -> Path:
    override = os.environ.get("SPHARMGRID_NCL_REFERENCE_DIR")
    if not override:
        raise RuntimeError(
            "SPHARMGRID_NCL_REFERENCE_DIR must point to normalized NCL "
            "reference outputs generated for this test run"
        )
    return Path(override)


def load_reference(
    grid_kind: GridKind, input_family: InputFamily
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    path = reference_directory() / f"{grid_kind}-{input_family}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing generated NCL reference output {path}; run the "
            "documented NCL workflow first"
        )
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{path} has no metadata entry")
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.array(archive[name], dtype=np.float64, copy=True)
            for name in archive.files
            if name != "metadata"
        }
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported NCL reference schema in {path}")
    if metadata.get("grid_kind") != grid_kind:
        raise ValueError(f"NCL reference grid metadata disagrees with {path}")
    if metadata.get("input_family") != input_family:
        raise ValueError(f"NCL reference input metadata disagrees with {path}")
    if metadata.get("latitude_order") != "ascending":
        raise ValueError(f"NCL reference is not in canonical latitude order: {path}")
    if input_family == "random" and int(metadata.get("random_seed", -1)) != SEED:
        raise ValueError(f"NCL random seed metadata disagrees with {path}")
    return arrays, metadata


def grid(kind: GridKind, latitude_order: LatitudeOrder = "ascending") -> sg.Grid:
    if kind == "gl":
        return sg.gaussian_grid(64, 128, latitude_order=latitude_order)
    return sg.clenshaw_curtis_grid(73, 144, latitude_order=latitude_order)


def _target_kind_from_key(key: str) -> GridKind | None:
    if not key.startswith("regrid_"):
        return None
    pieces = key.split("_")
    if len(pieces) < 3 or pieces[2] not in GRID_KINDS:
        raise ValueError(f"unrecognized regrid reference key: {key}")
    return cast(GridKind, pieces[2])


def data_array(
    arrays: dict[str, NDArray[np.float64]],
    key: str,
    source_grid: GridKind,
    *,
    latitude_order: LatitudeOrder = "ascending",
    target_grid: GridKind | None = None,
) -> xr.DataArray:
    resolved_target = target_grid or _target_kind_from_key(key)
    kind = resolved_target or source_grid
    target = resolved_target is not None
    target_latitude = f"target_lat_{kind}" if target else "lat"
    target_longitude = f"target_lon_{kind}" if target else "lon"
    values = arrays[key]
    coordinates = {
        "lat": arrays[target_latitude]
        if latitude_order == "ascending"
        else arrays[target_latitude][::-1],
        "lon": arrays[target_longitude],
    }
    if latitude_order == "descending":
        values = values[::-1, :]
    return xr.DataArray(values, dims=("lat", "lon"), coords=coordinates, name=key)


def expected_values(
    arrays: dict[str, NDArray[np.float64]],
    key: str,
    source_grid: GridKind,
    *,
    latitude_order: LatitudeOrder = "ascending",
    target_grid: GridKind | None = None,
) -> NDArray[np.float64]:
    values = arrays[key]
    if latitude_order == "descending":
        return values[::-1, :]
    return values
