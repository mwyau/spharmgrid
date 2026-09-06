"""CF-aware atmospheric variable discovery and output metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import xarray as xr

Quantity = Literal["u", "v", "vo", "d", "strf", "vp"]
ScalarSource = Literal[
    "vorticity", "divergence", "streamfunction", "velocity_potential"
]


@dataclass(frozen=True, slots=True)
class QuantityMetadata:
    """Canonical identity and CF metadata for one supported atmospheric field."""

    short_name: Quantity
    standard_name: str
    long_name: str
    units: str


QUANTITIES: dict[Quantity, QuantityMetadata] = {
    "u": QuantityMetadata("u", "eastward_wind", "Eastward wind", "m s-1"),
    "v": QuantityMetadata("v", "northward_wind", "Northward wind", "m s-1"),
    "vo": QuantityMetadata(
        "vo",
        "atmosphere_relative_vorticity",
        "Relative vorticity",
        "s-1",
    ),
    "d": QuantityMetadata("d", "divergence_of_wind", "Divergence of wind", "s-1"),
    "strf": QuantityMetadata(
        "strf",
        "atmosphere_horizontal_streamfunction",
        "Horizontal streamfunction",
        "m2 s-1",
    ),
    "vp": QuantityMetadata(
        "vp",
        "atmosphere_horizontal_velocity_potential",
        "Horizontal velocity potential",
        "m2 s-1",
    ),
}

_SOURCE_TO_QUANTITY: dict[ScalarSource, Quantity] = {
    "vorticity": "vo",
    "divergence": "d",
    "streamfunction": "strf",
    "velocity_potential": "vp",
}


def find_variable(
    dataset: xr.Dataset, quantity: Quantity, explicit: str | None = None
) -> xr.DataArray:
    """Resolve a Dataset variable by explicit name, exact CF name, then short name."""
    if explicit is not None:
        if explicit not in dataset.data_vars:
            raise ValueError(
                f"explicit {quantity!r} variable {explicit!r} is not a data variable"
            )
        return dataset[explicit]

    metadata = QUANTITIES[quantity]
    cf_matches = sorted(
        str(name)
        for name, variable in dataset.data_vars.items()
        if variable.attrs.get("standard_name") == metadata.standard_name
    )
    if cf_matches:
        return _unique_variable(cf_matches, quantity, "CF standard_name", dataset)
    if quantity in dataset.data_vars:
        return dataset[quantity]
    raise ValueError(
        f"could not identify {quantity!r}; pass its variable name explicitly, "
        f"set standard_name={metadata.standard_name!r}, or use canonical name "
        f"{quantity!r}"
    )


def try_find_variable(dataset: xr.Dataset, quantity: Quantity) -> xr.DataArray | None:
    """Resolve a variable when present, preserving ambiguity errors."""
    try:
        return find_variable(dataset, quantity)
    except ValueError as error:
        if str(error).startswith("could not identify"):
            return None
        raise


def identify_scalar_source(
    field: xr.DataArray,
    *,
    allowed: tuple[ScalarSource, ...],
    quantity: ScalarSource | None = None,
) -> ScalarSource:
    """Identify a scalar transform source from explicit semantics, CF, or name."""
    if quantity is not None:
        if quantity not in allowed:
            choices = ", ".join(allowed)
            raise ValueError(f"quantity must be one of: {choices}")
        return quantity

    matches = [
        source
        for source in allowed
        if field.attrs.get("standard_name")
        == QUANTITIES[_SOURCE_TO_QUANTITY[source]].standard_name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("scalar field has ambiguous CF source semantics")

    if field.name is not None:
        name = str(field.name)
        matches = [source for source in allowed if _SOURCE_TO_QUANTITY[source] == name]
        if len(matches) == 1:
            return matches[0]
    choices = ", ".join(allowed)
    raise ValueError(
        f"could not identify scalar source; pass quantity= with one of: {choices}"
    )


def output_metadata(quantity: Quantity) -> dict[str, str]:
    """Return the canonical CF-compatible attrs for an output quantity."""
    metadata = QUANTITIES[quantity]
    return {
        "standard_name": metadata.standard_name,
        "long_name": metadata.long_name,
        "units": metadata.units,
    }


def with_output_metadata(
    field: xr.DataArray, quantity: Quantity, name: str
) -> xr.DataArray:
    """Assign canonical semantic metadata without changing coordinate metadata."""
    result = field.copy(deep=False)
    result.name = name
    result.attrs = output_metadata(quantity)
    return result


def preserve_quantity_metadata(
    field: xr.DataArray, source: xr.DataArray
) -> xr.DataArray:
    """Preserve all data-variable attrs for filter/regrid operations."""
    result = field.copy(deep=False)
    result.name = source.name
    result.attrs = dict(source.attrs)
    return result


def wind_component_metadata(
    component: Literal["eastward", "northward"],
    kind: Literal["rotational", "divergent"],
) -> dict[str, str]:
    """Metadata for a component that has no exact CF standard name."""
    direction = "Eastward" if component == "eastward" else "Northward"
    return {"long_name": f"{direction} {kind} wind", "units": "m s-1"}


def gradient_metadata(
    source: xr.DataArray, component: Literal["eastward", "northward"]
) -> dict[str, str]:
    """Metadata for a physical horizontal gradient component."""
    direction = "Eastward" if component == "eastward" else "Northward"
    attrs = {"long_name": f"{direction} component of horizontal gradient"}
    units = source.attrs.get("units")
    if isinstance(units, str) and units:
        attrs["units"] = f"{units} m-1"
    return attrs


def inverse_gradient_metadata(
    eastward: xr.DataArray, northward: xr.DataArray
) -> dict[str, str]:
    """Metadata for a potential reconstructed from horizontal derivatives."""
    attrs = {
        "long_name": (
            "Scalar potential of the irrotational component of a horizontal vector"
        )
    }
    eastward_units = eastward.attrs.get("units")
    northward_units = northward.attrs.get("units")
    if (
        isinstance(eastward_units, str)
        and eastward_units == northward_units
        and eastward_units.endswith(" m-1")
    ):
        attrs["units"] = eastward_units.removesuffix(" m-1")
    return attrs


def operator_metadata(
    source: xr.DataArray, operation: Literal["laplacian", "inverse_laplacian"]
) -> dict[str, str]:
    """Metadata for scalar operators whose CF standard name depends on input."""
    readable = "Laplacian" if operation == "laplacian" else "Inverse Laplacian"
    source_name = source.attrs.get("long_name") or source.name or "field"
    attrs = {"long_name": f"{readable} of {source_name}"}
    units = source.attrs.get("units")
    if isinstance(units, str) and units:
        suffix = "m-2" if operation == "laplacian" else "m2"
        attrs["units"] = f"{units} {suffix}"
    return attrs


def vector_operator_metadata(
    source: xr.DataArray,
    component: Literal["eastward", "northward"],
    operation: Literal["laplacian", "inverse_laplacian"],
) -> dict[str, str]:
    """Metadata for a vector differential operator component."""
    direction = "Eastward" if component == "eastward" else "Northward"
    readable = (
        "vector Laplacian" if operation == "laplacian" else "inverse vector Laplacian"
    )
    source_name = source.attrs.get("long_name") or source.name or "vector field"
    attrs = {"long_name": f"{direction} component of {readable} of {source_name}"}
    units = source.attrs.get("units")
    if isinstance(units, str) and units:
        suffix = "m-2" if operation == "laplacian" else "m2"
        attrs["units"] = f"{units} {suffix}"
    return attrs


def _unique_variable(
    candidates: list[str], quantity: Quantity, source: str, dataset: xr.Dataset
) -> xr.DataArray:
    if len(candidates) == 1:
        return dataset[candidates[0]]
    joined = ", ".join(candidates)
    raise ValueError(f"ambiguous {quantity!r} variables from {source}: {joined}")
