"""Thin xarray ``.sg`` accessors delegating to the direct public functions."""

from __future__ import annotations

from typing import Literal

import xarray as xr

from .grids import Grid, detect_grid
from .kinematics import (
    divergence as calculate_divergence,
)
from .kinematics import (
    divergent_wind as calculate_divergent_wind,
)
from .kinematics import (
    helmholtz as calculate_helmholtz,
)
from .kinematics import (
    inverse_vector_laplacian as calculate_inverse_vector_laplacian,
)
from .kinematics import (
    kinematics as calculate_kinematics,
)
from .kinematics import (
    potentials as calculate_potentials,
)
from .kinematics import (
    rotational_wind as calculate_rotational_wind,
)
from .kinematics import (
    streamfunction as calculate_streamfunction,
)
from .kinematics import (
    vector_laplacian as calculate_vector_laplacian,
)
from .kinematics import (
    velocity_potential as calculate_velocity_potential,
)
from .kinematics import (
    vorticity as calculate_vorticity,
)
from .kinematics import (
    wind as calculate_wind,
)
from .metadata import Quantity, ScalarSource, find_variable, try_find_variable
from .operators import EARTH_RADIUS_M
from .operators import gradient as calculate_gradient
from .operators import inverse_gradient as calculate_inverse_gradient
from .operators import inverse_laplacian as calculate_inverse_laplacian
from .operators import laplacian as calculate_laplacian
from .regrid import regrid as calculate_regrid
from .regrid import regrid_vector as calculate_regrid_vector
from .spectral import SpectralRange
from .spectral import filter as calculate_filter


@xr.register_dataarray_accessor("sg")
class DataArrayAccessor:
    """Spherical-grid operations for one xarray field.

    The accessor contains no numerical implementation.  Each method calls the
    matching direct :mod:`spharmgrid` function, so both public styles follow
    the same coordinate and DUCC transform path.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def grid(self) -> Grid:
        """The detected supported GL or CC horizontal grid."""
        return detect_grid(self._obj)

    @property
    def grid_type(self) -> Literal["gl", "cc"]:
        """The lowercase detected grid family."""
        return self.grid.kind

    def filter(
        self,
        truncation: str | SpectralRange | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Filter this field by total wavenumber."""
        return calculate_filter(
            self._obj,
            truncation,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
            nthreads=nthreads,
        )

    def regrid(
        self,
        target_grid: Grid | xr.DataArray | xr.Dataset,
        truncation: str | SpectralRange | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Regrid this field, optionally filtering in the same transform cycle."""
        return calculate_regrid(
            self._obj,
            target_grid,
            truncation,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
            nthreads=nthreads,
        )

    def regrid_vector(
        self,
        v: xr.DataArray,
        target_grid: Grid | xr.DataArray | xr.Dataset,
        truncation: str | SpectralRange | None = None,
        *,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
        eastward: str = "u",
        northward: str = "v",
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Regrid this eastward component and ``v`` through one vector SHT."""
        return calculate_regrid_vector(
            self._obj,
            v,
            target_grid,
            truncation,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
            eastward=eastward,
            northward=northward,
            nthreads=nthreads,
        )

    def gradient(
        self,
        *,
        eastward: str = "gradient_eastward",
        northward: str = "gradient_northward",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Compute this scalar field's physical horizontal gradient."""
        return calculate_gradient(
            self._obj,
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def laplacian(
        self, *, radius: float = EARTH_RADIUS_M, nthreads: int | None = None
    ) -> xr.DataArray:
        """Compute this scalar field's physical spherical Laplacian."""
        return calculate_laplacian(self._obj, radius=radius, nthreads=nthreads)

    def inverse_laplacian(
        self, *, radius: float = EARTH_RADIUS_M, nthreads: int | None = None
    ) -> xr.DataArray:
        """Compute this field's zero-mean inverse spherical Laplacian."""
        return calculate_inverse_laplacian(self._obj, radius=radius, nthreads=nthreads)

    def inverse_gradient(
        self,
        northward: xr.DataArray,
        *,
        output: str | None = None,
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Treat this field as eastward gradient and recover its scalar potential."""
        return calculate_inverse_gradient(
            self._obj,
            northward,
            output=output,
            radius=radius,
            nthreads=nthreads,
        )

    def vorticity(
        self,
        v: xr.DataArray,
        *,
        output: str = "vo",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and compute relative vorticity."""
        return calculate_vorticity(
            self._obj, v, output=output, radius=radius, nthreads=nthreads
        )

    def divergence(
        self,
        v: xr.DataArray,
        *,
        output: str = "d",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and compute divergence."""
        return calculate_divergence(
            self._obj, v, output=output, radius=radius, nthreads=nthreads
        )

    def kinematics(
        self,
        v: xr.DataArray,
        *,
        vorticity: str = "vo",
        divergence: str = "d",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Treat this field as eastward wind and compute both diagnostics."""
        return calculate_kinematics(
            self._obj,
            v,
            vorticity=vorticity,
            divergence=divergence,
            radius=radius,
            nthreads=nthreads,
        )

    def streamfunction(
        self,
        v: xr.DataArray,
        *,
        output: str = "strf",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and calculate streamfunction."""
        return calculate_streamfunction(
            self._obj, v, output=output, radius=radius, nthreads=nthreads
        )

    def velocity_potential(
        self,
        v: xr.DataArray,
        *,
        output: str = "vp",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Treat this field as eastward wind and calculate velocity potential."""
        return calculate_velocity_potential(
            self._obj, v, output=output, radius=radius, nthreads=nthreads
        )

    def potentials(
        self,
        v: xr.DataArray,
        *,
        streamfunction: str = "strf",
        velocity_potential: str = "vp",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Treat this field as eastward wind and calculate both potentials."""
        return calculate_potentials(
            self._obj,
            v,
            streamfunction=streamfunction,
            velocity_potential=velocity_potential,
            radius=radius,
            nthreads=nthreads,
        )

    def helmholtz(
        self,
        v: xr.DataArray,
        *,
        divergent_eastward: str = "u_divergent",
        divergent_northward: str = "v_divergent",
        rotational_eastward: str = "u_rotational",
        rotational_northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Split this eastward wind and ``v`` into divergent and rotational wind."""
        return calculate_helmholtz(
            self._obj,
            v,
            divergent_eastward=divergent_eastward,
            divergent_northward=divergent_northward,
            rotational_eastward=rotational_eastward,
            rotational_northward=rotational_northward,
            radius=radius,
            nthreads=nthreads,
        )

    def vector_laplacian(
        self,
        v: xr.DataArray,
        *,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Apply the vector Laplacian to this eastward component and ``v``."""
        return calculate_vector_laplacian(
            self._obj,
            v,
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def inverse_vector_laplacian(
        self,
        v: xr.DataArray,
        *,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Apply the zero-mode-defined inverse vector Laplacian to this vector."""
        return calculate_inverse_vector_laplacian(
            self._obj,
            v,
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def rotational_wind(
        self,
        *,
        quantity: Literal["vorticity", "streamfunction"] | None = None,
        eastward: str = "u_rotational",
        northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Recover rotational wind from this vorticity or streamfunction field."""
        return calculate_rotational_wind(
            self._obj,
            quantity=quantity,
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def divergent_wind(
        self,
        *,
        quantity: Literal["divergence", "velocity_potential"] | None = None,
        eastward: str = "u_divergent",
        northward: str = "v_divergent",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Recover divergent wind from this divergence or potential field."""
        return calculate_divergent_wind(
            self._obj,
            quantity=quantity,
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def wind(
        self,
        second: xr.DataArray,
        *,
        source: Literal["vorticity_divergence", "potentials"] | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Reconstruct wind using this field as the first scalar source."""
        return calculate_wind(
            self._obj,
            second,
            source=source,
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )


@xr.register_dataset_accessor("sg")
class DatasetAccessor:
    """CF-aware wind operations for an xarray Dataset.

    Explicit names win; otherwise methods use exact CF ``standard_name``
    metadata and then the documented canonical names.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    @property
    def grid(self) -> Grid:
        """The Dataset's detected supported global grid."""
        return detect_grid(self._obj)

    @property
    def grid_type(self) -> Literal["gl", "cc"]:
        """The lowercase detected grid family."""
        return self.grid.kind

    def regrid_vector(
        self,
        target_grid: Grid | xr.DataArray | xr.Dataset,
        truncation: str | SpectralRange | None = None,
        *,
        u: str | None = None,
        v: str | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        taper: float | None = None,
        eastward: str = "u",
        northward: str = "v",
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find wind components and spectrally regrid them as one vector field."""
        return calculate_regrid_vector(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            target_grid,
            truncation,
            lmin=lmin,
            lmax=lmax,
            taper=taper,
            eastward=eastward,
            northward=northward,
            nthreads=nthreads,
        )

    def inverse_gradient(
        self,
        *,
        eastward: str = "gradient_eastward",
        northward: str = "gradient_northward",
        output: str | None = None,
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Recover a scalar potential from named horizontal-gradient components."""
        if eastward not in self._obj.data_vars or northward not in self._obj.data_vars:
            raise ValueError(
                "inverse_gradient requires explicit gradient component variables; "
                "pass eastward= and northward="
            )
        return calculate_inverse_gradient(
            self._obj[eastward],
            self._obj[northward],
            output=output,
            radius=radius,
            nthreads=nthreads,
        )

    def vorticity(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "vo",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Find wind components and compute relative vorticity."""
        return calculate_vorticity(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            output=output,
            radius=radius,
            nthreads=nthreads,
        )

    def divergence(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "d",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Find wind components and compute divergence."""
        return calculate_divergence(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            output=output,
            radius=radius,
            nthreads=nthreads,
        )

    def kinematics(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        vorticity: str = "vo",
        divergence: str = "d",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find wind components and compute vorticity plus divergence."""
        return calculate_kinematics(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            vorticity=vorticity,
            divergence=divergence,
            radius=radius,
            nthreads=nthreads,
        )

    def streamfunction(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "strf",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Find wind components and compute streamfunction."""
        return calculate_streamfunction(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            output=output,
            radius=radius,
            nthreads=nthreads,
        )

    def velocity_potential(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        output: str = "vp",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.DataArray:
        """Find wind components and compute velocity potential."""
        return calculate_velocity_potential(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            output=output,
            radius=radius,
            nthreads=nthreads,
        )

    def potentials(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        streamfunction: str = "strf",
        velocity_potential: str = "vp",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find wind components and compute both scalar potentials."""
        return calculate_potentials(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            streamfunction=streamfunction,
            velocity_potential=velocity_potential,
            radius=radius,
            nthreads=nthreads,
        )

    def helmholtz(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        divergent_eastward: str = "u_divergent",
        divergent_northward: str = "v_divergent",
        rotational_eastward: str = "u_rotational",
        rotational_northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find wind components and split them into divergent and rotational wind."""
        return calculate_helmholtz(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            divergent_eastward=divergent_eastward,
            divergent_northward=divergent_northward,
            rotational_eastward=rotational_eastward,
            rotational_northward=rotational_northward,
            radius=radius,
            nthreads=nthreads,
        )

    def vector_laplacian(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find wind components and apply the vector Laplacian."""
        return calculate_vector_laplacian(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def inverse_vector_laplacian(
        self,
        *,
        u: str | None = None,
        v: str | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find wind components and apply the inverse vector Laplacian."""
        return calculate_inverse_vector_laplacian(
            find_variable(self._obj, "u", u),
            find_variable(self._obj, "v", v),
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def rotational_wind(
        self,
        *,
        field: str | None = None,
        quantity: Literal["vorticity", "streamfunction"] | None = None,
        eastward: str = "u_rotational",
        northward: str = "v_rotational",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find a vorticity or streamfunction field and recover rotational wind."""
        selected, inferred = _find_single_source(
            self._obj,
            field,
            quantity,
            primary="vo",
            secondary="strf",
        )
        return calculate_rotational_wind(
            selected,
            quantity=_as_rotational_source(inferred),
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def divergent_wind(
        self,
        *,
        field: str | None = None,
        quantity: Literal["divergence", "velocity_potential"] | None = None,
        eastward: str = "u_divergent",
        northward: str = "v_divergent",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Find divergence or velocity potential and recover divergent wind."""
        selected, inferred = _find_single_source(
            self._obj,
            field,
            quantity,
            primary="d",
            secondary="vp",
        )
        return calculate_divergent_wind(
            selected,
            quantity=_as_divergent_source(inferred),
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )

    def wind(
        self,
        *,
        source: Literal["vorticity_divergence", "potentials"] | None = None,
        vorticity: str | None = None,
        divergence: str | None = None,
        streamfunction: str | None = None,
        velocity_potential: str | None = None,
        eastward: str = "u",
        northward: str = "v",
        radius: float = EARTH_RADIUS_M,
        nthreads: int | None = None,
    ) -> xr.Dataset:
        """Reconstruct wind from the Dataset's one complete scalar representation."""
        resolved = _resolve_dataset_wind_source(
            self._obj,
            source=source,
            vorticity=vorticity,
            divergence=divergence,
            streamfunction=streamfunction,
            velocity_potential=velocity_potential,
        )
        return calculate_wind(
            resolved[1],
            resolved[2],
            source=resolved[0],
            eastward=eastward,
            northward=northward,
            radius=radius,
            nthreads=nthreads,
        )


def _find_single_source(
    dataset: xr.Dataset,
    explicit: str | None,
    quantity: ScalarSource | None,
    *,
    primary: Literal["vo", "d"],
    secondary: Literal["strf", "vp"],
) -> tuple[xr.DataArray, ScalarSource | None]:
    if explicit is not None:
        if explicit not in dataset.data_vars:
            raise ValueError(f"explicit field {explicit!r} is not a data variable")
        return dataset[explicit], quantity
    if quantity is not None:
        lookups: dict[ScalarSource, Quantity] = {
            "vorticity": "vo",
            "streamfunction": "strf",
            "divergence": "d",
            "velocity_potential": "vp",
        }
        lookup = lookups.get(quantity)
        if lookup is None:
            choices = ", ".join(lookups)
            raise ValueError(f"quantity must be one of: {choices}")
        return find_variable(dataset, lookup), quantity
    first = try_find_variable(dataset, primary)
    second = try_find_variable(dataset, secondary)
    if first is not None and second is not None:
        raise ValueError(
            "both eligible scalar sources are present; pass field= or quantity="
        )
    if first is not None:
        return first, None
    if second is not None:
        return second, None
    raise ValueError("could not identify an eligible scalar source in the Dataset")


def _as_rotational_source(
    value: ScalarSource | None,
) -> Literal["vorticity", "streamfunction"] | None:
    if value is None:
        return None
    if value in {"vorticity", "streamfunction"}:
        return value
    raise ValueError("selected scalar source is not vorticity or streamfunction")


def _as_divergent_source(
    value: ScalarSource | None,
) -> Literal["divergence", "velocity_potential"] | None:
    if value is None:
        return None
    if value in {"divergence", "velocity_potential"}:
        return value
    raise ValueError("selected scalar source is not divergence or velocity potential")


def _resolve_dataset_wind_source(
    dataset: xr.Dataset,
    *,
    source: Literal["vorticity_divergence", "potentials"] | None,
    vorticity: str | None,
    divergence: str | None,
    streamfunction: str | None,
    velocity_potential: str | None,
) -> tuple[Literal["vorticity_divergence", "potentials"], xr.DataArray, xr.DataArray]:
    if source is not None and source not in {"vorticity_divergence", "potentials"}:
        raise ValueError("source must be 'vorticity_divergence' or 'potentials'")

    if source == "vorticity_divergence":
        if streamfunction is not None or velocity_potential is not None:
            raise ValueError(
                "source='vorticity_divergence' does not accept potential "
                "source-variable overrides"
            )
        have_vorticity = _find_dataset_source(dataset, "vo", vorticity)
        have_divergence = _find_dataset_source(dataset, "d", divergence)
        if have_vorticity is None or have_divergence is None:
            raise ValueError("wind source vorticity_divergence requires both vo and d")
        return source, have_vorticity, have_divergence

    if source == "potentials":
        if vorticity is not None or divergence is not None:
            raise ValueError(
                "source='potentials' does not accept vorticity/divergence "
                "source-variable overrides"
            )
        have_streamfunction = _find_dataset_source(dataset, "strf", streamfunction)
        have_velocity_potential = _find_dataset_source(
            dataset, "vp", velocity_potential
        )
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


def _find_dataset_source(
    dataset: xr.Dataset, quantity: Quantity, explicit: str | None
) -> xr.DataArray | None:
    """Resolve an explicitly selected source without inspecting other kinds."""
    if explicit is not None:
        return find_variable(dataset, quantity, explicit)
    return try_find_variable(dataset, quantity)
