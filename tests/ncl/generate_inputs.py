# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create the byte-stable scalar and wind inputs used by the NCL suite."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import xarray as xr

import spharmgrid as sg

SEED = 42
GRID_KINDS = ("gl", "cc")
INPUT_FAMILIES = ("analytic", "random")
GridKind = Literal["gl", "cc"]
ScalarTerm: TypeAlias = tuple[int, int, Literal["cos", "sin"], float]


def _grid(kind: GridKind) -> sg.Grid:
    if kind == "gl":
        return sg.gaussian_grid(64, 128, latitude_order="ascending")
    return sg.clenshaw_curtis_grid(73, 144, latitude_order="ascending")


def _target_grids() -> dict[GridKind, sg.Grid]:
    return {"gl": _grid("gl"), "cc": _grid("cc")}


def _associated_legendre(degree: int, m: int, x: np.ndarray) -> np.ndarray:
    """Return a Schmidt-normalized real associated Legendre factor.

    The recurrence is intentionally local to input construction.  It does not
    call spharmgrid's transform or spectral-mask implementation.
    """
    pmm = np.ones_like(x, dtype=np.float64)
    if m:
        odd_product = math.prod(range(1, 2 * m, 2))
        pmm = ((-1.0) ** m) * odd_product * (1.0 - x * x) ** (m / 2.0)
    if degree == m:
        value = pmm
    else:
        pm1 = (2 * m + 1) * x * pmm
        if degree == m + 1:
            value = pm1
        else:
            for current_degree in range(m + 2, degree + 1):
                current = (
                    (2 * current_degree - 1) * x * pm1 - (current_degree + m - 1) * pmm
                ) / (current_degree - m)
                pmm, pm1 = pm1, current
            value = pm1

    normalization = math.sqrt(
        (2 * degree + 1)
        / (4.0 * math.pi)
        * math.exp(math.lgamma(degree - m + 1) - math.lgamma(degree + m + 1))
    )
    return normalization * value


def _real_harmonic(
    latitude: np.ndarray,
    longitude: np.ndarray,
    degree: int,
    m: int,
    phase: Literal["cos", "sin"],
) -> np.ndarray:
    x = np.sin(latitude)[:, None]
    legendre = _associated_legendre(degree, m, x)
    angle = m * longitude[None, :]
    zonal = np.cos(angle) if phase == "cos" else np.sin(angle)
    return legendre * zonal


def _analytic_fields(grid: sg.Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = np.deg2rad(grid.latitude)
    longitude = np.deg2rad(grid.longitude)

    # These terms deliberately straddle l=5, l=21, l=42, m=10, and the
    # R21 diagonal.  Terms above l=42 verify that explicit truncations remove
    # them, while the l=30,m=22 and l=42,m=30 terms distinguish R21 from T42.
    scalar_terms: tuple[ScalarTerm, ...] = (
        (0, 0, "cos", 0.70),
        (1, 0, "cos", 0.31),
        (4, 2, "cos", 0.43),
        (5, 1, "sin", 0.37),
        (5, 5, "cos", -0.29),
        (10, 10, "sin", 0.23),
        (11, 11, "cos", -0.19),
        (20, 3, "cos", 0.17),
        (21, 0, "cos", 0.21),
        (21, 21, "sin", 0.15),
        (22, 0, "cos", -0.18),
        (22, 10, "cos", 0.16),
        (22, 11, "sin", 0.14),
        (30, 22, "cos", 0.055),
        (41, 9, "cos", 0.13),
        (42, 0, "cos", 0.12),
        (42, 10, "sin", -0.11),
        (42, 11, "cos", 0.10),
        (42, 30, "sin", -0.05),
        (43, 0, "cos", 0.09),
        (43, 11, "sin", -0.08),
        (50, 25, "cos", 0.07),
        (50, 50, "sin", 0.06),
    )

    scalar = np.zeros((grid.nlat, grid.nlon), dtype=np.float64)
    for degree, order, phase, amplitude in scalar_terms:
        scalar += amplitude * _real_harmonic(latitude, longitude, degree, order, phase)

    # The cos(latitude) factor makes both geographic components regular at
    # the CC poles.  The same broadband harmonic terms are used in distinct
    # mixtures so vector diagnostics exercise both E and B content.
    cosine_latitude = np.cos(latitude)[:, None]
    u = np.zeros_like(scalar)
    v = np.zeros_like(scalar)
    for index, (degree, order, phase, amplitude) in enumerate(scalar_terms):
        basis = _real_harmonic(latitude, longitude, degree, order, phase)
        u += (0.8 + 0.03 * index) * amplitude * basis
        v += (-0.6 + 0.02 * index) * amplitude * basis
    u *= cosine_latitude
    v *= cosine_latitude
    return scalar, u, v


def _random_fields(
    grid: sg.Grid,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (grid.nlat, grid.nlon)
    return (
        rng.standard_normal(shape, dtype=np.float64),
        rng.standard_normal(shape, dtype=np.float64),
        rng.standard_normal(shape, dtype=np.float64),
    )


def build_input(kind: GridKind, family: Literal["analytic", "random"]) -> xr.Dataset:
    grid = _grid(kind)
    target_grids = _target_grids()
    if family == "analytic":
        scalar, u, v = _analytic_fields(grid)
    else:
        rng = np.random.default_rng(SEED)
        scalar, u, v = _random_fields(grid, rng)
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    dataset = xr.Dataset(
        {
            "scalar": xr.DataArray(scalar, dims=("lat", "lon"), coords=coordinates),
            "u": xr.DataArray(u, dims=("lat", "lon"), coords=coordinates),
            "v": xr.DataArray(v, dims=("lat", "lon"), coords=coordinates),
            "target_lat_gl": xr.DataArray(
                target_grids["gl"].latitude, dims=("target_lat_gl",)
            ),
            "target_lon_gl": xr.DataArray(
                target_grids["gl"].longitude, dims=("target_lon_gl",)
            ),
            "target_lat_cc": xr.DataArray(
                target_grids["cc"].latitude, dims=("target_lat_cc",)
            ),
            "target_lon_cc": xr.DataArray(
                target_grids["cc"].longitude, dims=("target_lon_cc",)
            ),
        },
        attrs={
            "schema_version": "1",
            "grid_kind": kind,
            "input_family": family,
            "latitude_order": "ascending",
            "longitude_convention": "0_to_360_noncyclic",
            "random_seed": str(SEED) if family == "random" else "not_applicable",
        },
    )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid", choices=GRID_KINDS)
    parser.add_argument("--family", choices=INPUT_FAMILIES)
    args = parser.parse_args()
    kinds = (args.grid,) if args.grid else GRID_KINDS
    families = (args.family,) if args.family else INPUT_FAMILIES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        for family in families:
            dataset = build_input(kind, family)
            path = args.output_dir / f"input-{kind}-{family}.nc"
            dataset.to_netcdf(path, engine="h5netcdf")
            print(path)


if __name__ == "__main__":
    main()
