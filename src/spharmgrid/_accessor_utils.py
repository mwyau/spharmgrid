# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared CF-aware source resolution for the Xarray accessors."""

from __future__ import annotations

from typing import Literal

import xarray as xr

from ._kinematics_types import ScalarSource, WindSource
from .metadata import Quantity, find_variable, try_find_variable


def find_single_source(
    dataset: xr.Dataset,
    explicit: str | None,
    source: ScalarSource | None,
    *,
    primary: Literal["vo", "d"],
    secondary: Literal["strf", "vp"],
) -> tuple[xr.DataArray, ScalarSource | None]:
    """Resolve one scalar source for a Dataset wind operation."""
    if explicit is not None:
        if explicit not in dataset.data_vars:
            raise ValueError(f"explicit field {explicit!r} is not a data variable")
        return dataset[explicit], source
    if source is not None:
        lookups: dict[ScalarSource, Quantity] = {
            "vorticity": "vo",
            "streamfunction": "strf",
            "divergence": "d",
            "velocity_potential": "vp",
        }
        lookup = lookups.get(source)
        if lookup is None:
            choices = ", ".join(lookups)
            raise ValueError(f"source must be one of: {choices}")
        return find_variable(dataset, lookup), source
    first = try_find_variable(dataset, primary)
    second = try_find_variable(dataset, secondary)
    if first is not None and second is not None:
        raise ValueError(
            "both eligible scalar sources are present; pass field= or source="
        )
    if first is not None:
        return first, None
    if second is not None:
        return second, None
    raise ValueError("could not identify an eligible scalar source in the Dataset")


def as_rotational_source(
    value: ScalarSource | None,
) -> Literal["vorticity", "streamfunction"] | None:
    """Validate a source selected for rotational wind reconstruction."""
    if value is None:
        return None
    if value in {"vorticity", "streamfunction"}:
        return value
    raise ValueError("selected scalar source is not vorticity or streamfunction")


def as_divergent_source(
    value: ScalarSource | None,
) -> Literal["divergence", "velocity_potential"] | None:
    """Validate a source selected for divergent wind reconstruction."""
    if value is None:
        return None
    if value in {"divergence", "velocity_potential"}:
        return value
    raise ValueError("selected scalar source is not divergence or velocity potential")


def resolve_dataset_wind_source(
    dataset: xr.Dataset,
    *,
    source: WindSource | None,
    vorticity: str | None,
    divergence: str | None,
    streamfunction: str | None,
    velocity_potential: str | None,
) -> tuple[WindSource, xr.DataArray, xr.DataArray]:
    """Resolve one complete Dataset representation for wind reconstruction."""
    if source is not None and source not in {"vorticity_divergence", "potentials"}:
        raise ValueError("source must be 'vorticity_divergence' or 'potentials'")

    if source == "vorticity_divergence":
        if streamfunction is not None or velocity_potential is not None:
            raise ValueError(
                "source='vorticity_divergence' does not accept potential "
                "source-variable overrides"
            )
        have_vorticity = find_dataset_source(dataset, "vo", vorticity)
        have_divergence = find_dataset_source(dataset, "d", divergence)
        if have_vorticity is None or have_divergence is None:
            raise ValueError("wind source vorticity_divergence requires both vo and d")
        return source, have_vorticity, have_divergence

    if source == "potentials":
        if vorticity is not None or divergence is not None:
            raise ValueError(
                "source='potentials' does not accept vorticity/divergence "
                "source-variable overrides"
            )
        have_streamfunction = find_dataset_source(dataset, "strf", streamfunction)
        have_velocity_potential = find_dataset_source(dataset, "vp", velocity_potential)
        if have_streamfunction is None or have_velocity_potential is None:
            raise ValueError("wind source potentials requires both strf and vp")
        return source, have_streamfunction, have_velocity_potential

    have_vorticity = (
        try_find_variable(dataset, "vo")
        if vorticity is None
        else find_variable(dataset, "vo", vorticity)
    )
    have_divergence = (
        try_find_variable(dataset, "d")
        if divergence is None
        else find_variable(dataset, "d", divergence)
    )
    have_streamfunction = (
        try_find_variable(dataset, "strf")
        if streamfunction is None
        else find_variable(dataset, "strf", streamfunction)
    )
    have_velocity_potential = (
        try_find_variable(dataset, "vp")
        if velocity_potential is None
        else find_variable(dataset, "vp", velocity_potential)
    )
    vd_complete = have_vorticity is not None and have_divergence is not None
    potential_complete = (
        have_streamfunction is not None and have_velocity_potential is not None
    )
    if vd_complete and potential_complete:
        raise ValueError(
            "both vorticity/divergence and potentials are present; pass source="
        )
    if vd_complete:
        assert have_vorticity is not None
        assert have_divergence is not None
        return "vorticity_divergence", have_vorticity, have_divergence
    if potential_complete:
        assert have_streamfunction is not None
        assert have_velocity_potential is not None
        return "potentials", have_streamfunction, have_velocity_potential
    raise ValueError(
        "Dataset has no complete vorticity/divergence or potentials source"
    )


def find_dataset_source(
    dataset: xr.Dataset, quantity: Quantity, explicit: str | None
) -> xr.DataArray | None:
    """Resolve an explicitly selected source without inspecting other kinds."""
    if explicit is not None:
        return find_variable(dataset, quantity, explicit)
    return try_find_variable(dataset, quantity)


__all__ = [
    "as_divergent_source",
    "as_rotational_source",
    "find_dataset_source",
    "find_single_source",
    "resolve_dataset_wind_source",
]
