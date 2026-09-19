# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalize NCL NetCDF output for the parity tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from generate_inputs import GRID_KINDS, INPUT_FAMILIES, RANDOM_SEED

SCHEMA_VERSION = 1
DOMAINS = ("T42", "T5_42", "T42x10", "R21")
GRID_SHAPES = {"gl": (64, 128), "cc": (73, 144)}
COMMON_FIELDS = (
    "gradient_eastward",
    "gradient_northward",
    "inverse_gradient",
    "laplacian",
    "inverse_laplacian",
    "vorticity",
    "divergence",
    "kinematic_vorticity",
    "kinematic_divergence",
    "streamfunction",
    "velocity_potential",
    "rotational_u",
    "rotational_v",
    "divergent_u",
    "divergent_v",
    "wind_vo_d_u",
    "wind_vo_d_v",
    "wind_potentials_u",
    "wind_potentials_v",
    "helmholtz_u_rotational",
    "helmholtz_v_rotational",
    "helmholtz_u_divergent",
    "helmholtz_v_divergent",
    "vector_laplacian_u",
    "vector_laplacian_v",
    "inverse_vector_laplacian_u",
    "inverse_vector_laplacian_v",
)


def _required_attr(attrs: Mapping[str, object], name: str) -> object:
    value = attrs.get(name)
    if value is None:
        raise ValueError(f"NCL output is missing attribute {name!r}")
    return value


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _as_double(dataset: xr.Dataset, name: str) -> np.ndarray:
    if name not in dataset:
        raise ValueError(f"NCL output is missing variable {name!r}")
    values = np.asarray(dataset[name].values)
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} is {values.dtype}, expected double precision")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return np.array(values, dtype=np.float64, copy=True)


def _ascending_coordinate(dataset: xr.Dataset, name: str) -> tuple[np.ndarray, bool]:
    values = _as_double(dataset, name)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate")
    differences = np.diff(values)
    if np.all(differences > 0.0):
        return values, False
    if np.all(differences < 0.0):
        return values[::-1].copy(), True
    raise ValueError(f"{name} is not strictly monotonic")


def _field(
    dataset: xr.Dataset,
    name: str,
    *,
    shape: tuple[int, int],
    reverse_latitude: bool,
) -> np.ndarray:
    values = _as_double(dataset, name)
    if values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}, expected {shape}")
    if reverse_latitude:
        values = values[::-1, :]
    return values


def _manifest(grid_kind: str) -> dict[str, dict[str, Any]]:
    methods = {
        "gradient_eastward": "gradsg (GL) or gradsf (CC)",
        "gradient_northward": "gradsg (GL) or gradsf (CC)",
        "inverse_gradient": "igradsg (GL) or igradsf (CC)",
        "laplacian": "lapsg (GL) or lapsf (CC)",
        "inverse_laplacian": "ilapsg (GL) or ilapsf (CC)",
        "vorticity": "uv2vrg (GL) or uv2vrf (CC)",
        "divergence": "uv2dvg (GL) or uv2dvf (CC)",
        "kinematic_vorticity": "uv2vrdvg (GL) or uv2vrdvf (CC)",
        "kinematic_divergence": "uv2vrdvg (GL) or uv2vrdvf (CC)",
        "streamfunction": "uv2sfvpg (GL) or uv2sfvpf (CC)",
        "velocity_potential": "uv2sfvpg (GL) or uv2sfvpf (CC)",
        "rotational_u": "vr2uvg (GL) or vr2uvf (CC)",
        "rotational_v": "vr2uvg (GL) or vr2uvf (CC)",
        "divergent_u": "dv2uvg (GL) or dv2uvf (CC)",
        "divergent_v": "dv2uvg (GL) or dv2uvf (CC)",
        "wind_vo_d_u": "vrdv2uvg (GL) or vrdv2uvf (CC)",
        "wind_vo_d_v": "vrdv2uvg (GL) or vrdv2uvf (CC)",
        "wind_potentials_u": "sfvp2uvg (GL) or sfvp2uvf (CC)",
        "wind_potentials_v": "sfvp2uvg (GL) or sfvp2uvf (CC)",
        "helmholtz_u_rotational": "vr2uvg (GL) or vr2uvf (CC)",
        "helmholtz_v_rotational": "vr2uvg (GL) or vr2uvf (CC)",
        "helmholtz_u_divergent": "dv2uvg (GL) or dv2uvf (CC)",
        "helmholtz_v_divergent": "dv2uvg (GL) or dv2uvf (CC)",
        "vector_laplacian_u": "lapvg (GL) or lapvf (CC)",
        "vector_laplacian_v": "lapvg (GL) or lapvf (CC)",
        "inverse_vector_laplacian_u": "ilapvg (GL) or ilapvf (CC)",
        "inverse_vector_laplacian_v": "ilapvg (GL) or ilapvf (CC)",
    }
    manifest: dict[str, dict[str, Any]] = {}
    for name in COMMON_FIELDS:
        manifest[name] = {
            "operation": name,
            "source_grid": grid_kind,
            "truncation": None,
            "taper": None,
            "ncl_reference": methods[name],
        }
    manifest["filter_full_hard"] = {
        "operation": "filter",
        "source_grid": grid_kind,
        "truncation": None,
        "taper": None,
        "ncl_reference": "shaec/shsec (CC) or shagc/shsgc (GL), full bandwidth",
    }
    for domain in DOMAINS:
        for suffix, taper in (("hard", None), ("taper", 0.1)):
            name = f"filter_{domain}_{suffix}"
            manifest[name] = {
                "operation": "filter",
                "source_grid": grid_kind,
                "truncation": domain.replace("_", "-", 1),
                "taper": taper,
                "ncl_reference": "coefficient mask + exp_tapersh_wgts + "
                "shaec/shsec (CC) or shagc/shsgc (GL)",
            }
    for target_kind in GRID_KINDS:
        for domain in ("full", *DOMAINS):
            suffixes = ("hard",) if domain == "full" else ("hard", "taper")
            for suffix in suffixes:
                taper = None if suffix == "hard" else 0.1
                truncation = None if domain == "full" else domain.replace("_", "-", 1)
                for operation in ("scalar", "vector_u", "vector_v"):
                    name = f"regrid_{operation}_{target_kind}_{domain}_{suffix}"
                    manifest[name] = {
                        "operation": (
                            "regrid" if operation == "scalar" else "regrid_vector"
                        ),
                        "source_grid": grid_kind,
                        "target_grid": target_kind,
                        "truncation": truncation,
                        "taper": taper,
                        "ncl_reference": (
                            "analysis/mask/synthesis using "
                            "shaec/shsec, shagc/shsgc, vhaec/vhsec, or "
                            "vhagc/vhsgc"
                        ),
                    }
    return manifest


def _normalize_one(
    ncl_path: Path,
    input_path: Path,
    output_path: Path,
) -> None:
    with (
        xr.open_dataset(ncl_path, engine="h5netcdf") as ncl,
        xr.open_dataset(input_path, engine="h5netcdf") as source,
    ):
        attrs = ncl.attrs
        if str(attrs.get("schema_version")) != "1":
            raise ValueError(f"unsupported NCL schema in {ncl_path}: {attrs}")
        grid_kind = str(attrs.get("grid_kind"))
        family = str(attrs.get("input_family"))
        if grid_kind not in GRID_KINDS or family not in INPUT_FAMILIES:
            raise ValueError(f"invalid NCL identity in {ncl_path}: {attrs}")
        if str(_required_attr(attrs, "ncl_version")) != "6.6.2":
            raise ValueError("NCL reference output must be generated by NCL 6.6.2")
        nlat, nlon = GRID_SHAPES[grid_kind]
        shape = (nlat, nlon)

        latitude, reverse_latitude = _ascending_coordinate(ncl, "lat")
        longitude, reverse_longitude = _ascending_coordinate(ncl, "lon")
        if reverse_longitude:
            raise ValueError("NCL source longitude must already be canonical ascending")
        target_coordinates: dict[str, np.ndarray] = {}
        target_reversals: dict[str, bool] = {}
        for target_kind in GRID_KINDS:
            target_lat, target_reverse = _ascending_coordinate(
                ncl, f"target_lat_{target_kind}"
            )
            target_lon, target_lon_reverse = _ascending_coordinate(
                ncl, f"target_lon_{target_kind}"
            )
            if target_lon_reverse:
                raise ValueError("NCL target longitude must be canonical ascending")
            target_coordinates[f"target_lat_{target_kind}"] = target_lat
            target_coordinates[f"target_lon_{target_kind}"] = target_lon
            target_reversals[target_kind] = target_reverse

        expected_inputs = {"scalar": "scalar_input", "u": "u_input", "v": "v_input"}
        arrays: dict[str, np.ndarray] = {
            "lat": latitude,
            "lon": longitude,
            **target_coordinates,
        }
        for normalized_name, ncl_name in expected_inputs.items():
            values = _field(
                ncl,
                ncl_name,
                shape=shape,
                reverse_latitude=reverse_latitude,
            )
            source_values = np.asarray(source[normalized_name].values, dtype=np.float64)
            np.testing.assert_array_equal(values, source_values)
            arrays[normalized_name] = values

        for name in COMMON_FIELDS:
            arrays[name] = _field(
                ncl,
                name,
                shape=shape,
                reverse_latitude=reverse_latitude,
            )
        arrays["taper_weights"] = _as_double(ncl, "taper_weights")
        if arrays["taper_weights"].shape != (43,):
            raise ValueError("taper_weights must contain degrees 0 through 42")

        for domain in ("full", *DOMAINS):
            suffixes = ("hard",) if domain == "full" else ("hard", "taper")
            for suffix in suffixes:
                filter_name = f"filter_{domain}_{suffix}"
                arrays[filter_name] = _field(
                    ncl,
                    filter_name,
                    shape=shape,
                    reverse_latitude=reverse_latitude,
                )
                for target_kind in GRID_KINDS:
                    target_shape = GRID_SHAPES[target_kind]
                    for operation in ("scalar", "vector_u", "vector_v"):
                        name = f"regrid_{operation}_{target_kind}_{domain}_{suffix}"
                        arrays[name] = _field(
                            ncl,
                            name,
                            shape=target_shape,
                            reverse_latitude=target_reversals[target_kind],
                        )

        if family == "random" and str(source.attrs.get("random_seed")) != str(
            RANDOM_SEED
        ):
            raise ValueError("random input seed metadata does not match the fixed seed")
        if str(_required_attr(attrs, "random_seed")) != str(
            source.attrs.get("random_seed")
        ):
            raise ValueError("NCL output seed metadata does not match the input")
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "grid_kind": grid_kind,
            "input_family": family,
            "ncl_version": str(_required_attr(attrs, "ncl_version")),
            "reference": str(_required_attr(attrs, "reference")),
            "latitude_order": "ascending",
            "longitude_convention": str(_required_attr(attrs, "longitude_convention")),
            "earth_radius_m": float(
                cast(float, _required_attr(attrs, "earth_radius_m"))
            ),
            "taper": float(cast(float, _required_attr(attrs, "taper"))),
            "taper_mode": float(cast(float, _required_attr(attrs, "taper_mode"))),
            "random_seed": str(source.attrs.get("random_seed")),
            "input_sha256": _array_digest(
                arrays["lat"],
                arrays["lon"],
                arrays["scalar"],
                arrays["u"],
                arrays["v"],
            ),
            "array_keys": sorted(arrays),
            "manifest": _manifest(grid_kind),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **cast(dict[str, Any], arrays),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ncl-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for grid_kind in GRID_KINDS:
        for family in INPUT_FAMILIES:
            _normalize_one(
                args.ncl_output_dir / f"ncl-{grid_kind}-{family}.nc",
                args.input_dir / f"input-{grid_kind}-{family}.nc",
                args.output_dir / f"{grid_kind}-{family}.npz",
            )


if __name__ == "__main__":
    main()
