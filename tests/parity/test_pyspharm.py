"""Optional GL/CC parity checks against pyspharm-syl/SPHEREPACK."""

from __future__ import annotations

from typing import Literal, Protocol, cast

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import scalar_field, solid_body_wind

pytestmark = pytest.mark.parity
spharm = pytest.importorskip(
    "spharm",
    reason="run with uv --group parity on a supported Python version",
)

# pyspharm-syl exposes synthesized maps at float32 precision.  These absolute
# tolerances are calibrated from the maximum errors of the parity fixtures.
_SCALAR_MAP_ATOL = 1.0e-6
_VECTOR_MAP_ATOL = 6.0e-6
_GRADIENT_ATOL = 5.0e-13
_KINEMATIC_ATOL = 3.0e-12
_WIND_POTENTIAL_ATOL = 2.5e1
_INVERSE_GRADIENT_POTENTIAL_ATOL = 1.2e1
_VECTOR_LAPLACIAN_ATOL = 1.0e-17


class _SpharmTransform(Protocol):
    """Subset of pyspharm-syl's SPHEREPACK wrapper used in this suite."""

    def grdtospec(
        self, values: np.ndarray, ntrunc: int | None = None
    ) -> np.ndarray: ...

    def spectogrd(self, coefficients: np.ndarray) -> np.ndarray: ...

    def getgrad(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def getvrtdivspec(
        self, u: np.ndarray, v: np.ndarray, ntrunc: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def getpsichi(
        self, u: np.ndarray, v: np.ndarray, ntrunc: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def getuv(
        self, vorticity: np.ndarray, divergence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _gaussian_grid() -> sg.Grid:
    return sg.gaussian_grid(16, 36, latitude_order="ascending")


def _reference_transform() -> _SpharmTransform:
    return cast(
        _SpharmTransform,
        spharm.Spharmt(
            36,
            16,
            rsphere=sg.EARTH_RADIUS_M,
            gridtype="gaussian",
            legfunc="stored",
        ),
    )


def _north_to_south(values: np.ndarray) -> np.ndarray:
    return values[::-1, :]


def _smooth_vector_field(grid: sg.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Construct low-degree E/B content without spharmgrid transforms."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    sine = np.sin(latitude)
    cosine = np.cos(latitude)
    u = (
        10.0 * cosine
        + 3.0 * sine * cosine * np.sin(2.0 * longitude)
        - 4.0 * cosine * np.sin(2.0 * longitude)
    )
    v = (
        7.0 * cosine
        + 3.0 * cosine * np.cos(2.0 * longitude)
        - 4.0 * sine * cosine * np.cos(2.0 * longitude)
    )
    return u, v


def _pure_divergent_vector_field(grid: sg.Grid) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient of a degree-one scalar potential."""
    latitude = np.deg2rad(grid.latitude)[:, None]
    eastward = np.zeros((grid.nlat, grid.nlon))
    northward = 7.0 * np.cos(latitude) * np.ones((1, grid.nlon))
    return eastward, northward


def _dataarray(values: np.ndarray, grid: sg.Grid, name: str) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name=name,
    )


def test_gaussian_scalar_filter_gradient_and_regrid_match_pyspharm() -> None:
    grid = _gaussian_grid()
    source = scalar_field(grid)
    reference = _reference_transform()
    source_north_to_south = _north_to_south(np.asarray(source.values, dtype=np.float64))
    coefficients = reference.grdtospec(source_north_to_south, ntrunc=7)

    filtered = sg.filter(source, "T7")
    reference_filtered = reference.spectogrd(coefficients)
    reference_east, reference_north = reference.getgrad(coefficients)
    target = sg.gaussian_grid(12, 24, latitude_order="ascending")
    target_reference = spharm.Spharmt(
        target.nlon,
        target.nlat,
        rsphere=sg.EARTH_RADIUS_M,
        gridtype="gaussian",
        legfunc="stored",
    )
    reference_regridded = spharm.regrid(
        reference, target_reference, source_north_to_south, ntrunc=7
    )
    gradient = sg.gradient(source)
    regridded = sg.regrid(source, target, spectral="T7")

    np.testing.assert_allclose(
        _north_to_south(np.asarray(filtered.values)),
        reference_filtered,
        rtol=0.0,
        atol=_SCALAR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_eastward.values)),
        reference_east,
        rtol=0.0,
        atol=_GRADIENT_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_northward.values)),
        reference_north,
        rtol=0.0,
        atol=_GRADIENT_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(regridded.values)),
        reference_regridded,
        rtol=0.0,
        atol=_SCALAR_MAP_ATOL,
    )


def test_regular_cc_scalar_filter_gradient_and_regrid_match_pyspharm() -> None:
    """Compare CC sampling to SPHEREPACK's pole-including regular grid."""
    grid = sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending")
    source = scalar_field(grid)
    reference = cast(
        _SpharmTransform,
        spharm.Spharmt(
            grid.nlon,
            grid.nlat,
            rsphere=sg.EARTH_RADIUS_M,
            gridtype="regular",
            legfunc="stored",
        ),
    )
    source_north_to_south = _north_to_south(np.asarray(source.values, dtype=np.float64))
    coefficients = reference.grdtospec(source_north_to_south, ntrunc=7)
    target = sg.clenshaw_curtis_grid(13, 24, latitude_order="ascending")
    target_reference = spharm.Spharmt(
        target.nlon,
        target.nlat,
        rsphere=sg.EARTH_RADIUS_M,
        gridtype="regular",
        legfunc="stored",
    )

    filtered = sg.filter(source, "T7")
    gradient = sg.gradient(source)
    regridded = sg.regrid(source, target, spectral="T7")
    reference_filtered = reference.spectogrd(coefficients)
    reference_east, reference_north = reference.getgrad(coefficients)
    reference_regridded = spharm.regrid(
        reference,
        target_reference,
        source_north_to_south,
        ntrunc=7,
    )

    # The regular-grid wrapper returns scalar maps at float32 precision; the
    # measured cross-backend field error is below 5e-7.
    np.testing.assert_allclose(
        _north_to_south(np.asarray(filtered.values)),
        reference_filtered,
        rtol=0.0,
        atol=_SCALAR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_eastward.values)),
        reference_east,
        rtol=0.0,
        atol=_GRADIENT_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_northward.values)),
        reference_north,
        rtol=0.0,
        atol=_GRADIENT_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(regridded.values)),
        reference_regridded,
        rtol=0.0,
        atol=_SCALAR_MAP_ATOL,
    )


@pytest.mark.parametrize(
    ("kind", "gridtype"),
    [
        ("gaussian", "gaussian"),
        ("cc", "regular"),
    ],
)
def test_wind_kinematics_potentials_and_inverse_match_pyspharm(
    kind: Literal["cc", "gaussian"],
    gridtype: Literal["gaussian", "regular"],
) -> None:
    grid = (
        _gaussian_grid()
        if kind == "gaussian"
        else sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending")
    )
    u, v = solid_body_wind(grid)
    # Include a degree-one divergent component so SPHEREPACK's potential
    # comparison has a resolved signal rather than testing only its
    # float32-level zero-divergence residual.
    v = v + (7.0 * np.cos(np.deg2rad(grid.latitude))[:, None] * np.ones((1, grid.nlon)))
    reference = (
        _reference_transform()
        if gridtype == "gaussian"
        else cast(
            _SpharmTransform,
            spharm.Spharmt(
                grid.nlon,
                grid.nlat,
                rsphere=sg.EARTH_RADIUS_M,
                gridtype=gridtype,
                legfunc="stored",
            ),
        )
    )
    reference_u = _north_to_south(np.asarray(u.values, dtype=np.float64))
    reference_v = _north_to_south(np.asarray(v.values, dtype=np.float64))
    reference_vorticity, reference_divergence = reference.getvrtdivspec(
        reference_u, reference_v, ntrunc=7
    )
    reference_vo = reference.spectogrd(reference_vorticity)
    reference_d = reference.spectogrd(reference_divergence)
    reference_psi, reference_chi = reference.getpsichi(
        reference_u, reference_v, ntrunc=7
    )
    reference_u_reconstructed, reference_v_reconstructed = reference.getuv(
        reference_vorticity, reference_divergence
    )

    diagnostics = sg.kinematics(u, v)
    potential = sg.potentials(u, v)
    reconstructed = sg.wind(diagnostics.vo, diagnostics.d)

    np.testing.assert_allclose(
        _north_to_south(np.asarray(diagnostics.vo.values)),
        reference_vo,
        rtol=0.0,
        atol=_KINEMATIC_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(diagnostics.d.values)),
        reference_d,
        rtol=0.0,
        atol=_KINEMATIC_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(potential.strf.values)),
        reference_psi,
        rtol=0.0,
        atol=_WIND_POTENTIAL_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(potential.vp.values)),
        reference_chi,
        rtol=0.0,
        atol=_WIND_POTENTIAL_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(reconstructed.u.values)),
        reference_u_reconstructed,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(reconstructed.v.values)),
        reference_v_reconstructed,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )


@pytest.mark.parametrize(
    ("kind", "gridtype"),
    [
        ("gaussian", "gaussian"),
        ("cc", "regular"),
    ],
)
def test_vector_sht_suite_matches_independent_spherepack(
    kind: Literal["cc", "gaussian"],
    gridtype: Literal["gaussian", "regular"],
) -> None:
    """Compare vector analysis/synthesis-derived Phase-2 operations directly."""
    grid = (
        _gaussian_grid()
        if kind == "gaussian"
        else sg.clenshaw_curtis_grid(17, 36, latitude_order="ascending")
    )
    target = (
        sg.gaussian_grid(12, 24, latitude_order="ascending")
        if kind == "gaussian"
        else sg.clenshaw_curtis_grid(13, 24, latitude_order="ascending")
    )
    reference = (
        _reference_transform()
        if gridtype == "gaussian"
        else cast(
            _SpharmTransform,
            spharm.Spharmt(
                grid.nlon,
                grid.nlat,
                rsphere=sg.EARTH_RADIUS_M,
                gridtype="regular",
                legfunc="stored",
            ),
        )
    )
    target_reference = cast(
        _SpharmTransform,
        spharm.Spharmt(
            target.nlon,
            target.nlat,
            rsphere=sg.EARTH_RADIUS_M,
            gridtype=gridtype,
            legfunc="stored",
        ),
    )
    u_values, v_values = _smooth_vector_field(grid)
    u = _dataarray(u_values, grid, "u")
    v = _dataarray(v_values, grid, "v")
    reference_u = _north_to_south(u_values)
    reference_v = _north_to_south(v_values)
    regrid_vorticity, regrid_divergence = reference.getvrtdivspec(
        reference_u,
        reference_v,
        ntrunc=7,
    )
    reference_target_u, reference_target_v = target_reference.getuv(
        regrid_vorticity,
        regrid_divergence,
    )
    reference_vorticity, reference_divergence = reference.getvrtdivspec(
        reference_u,
        reference_v,
        ntrunc=15,
    )
    reference_divergent_u, reference_divergent_v = reference.getuv(
        np.zeros_like(reference_vorticity),
        reference_divergence,
    )
    reference_rotational_u, reference_rotational_v = reference.getuv(
        reference_vorticity,
        np.zeros_like(reference_divergence),
    )
    inverse_gradient_u, inverse_gradient_v = _pure_divergent_vector_field(grid)
    reference_inverse_gradient_u = _north_to_south(inverse_gradient_u)
    reference_inverse_gradient_v = _north_to_south(inverse_gradient_v)
    _, inverse_gradient_divergence = reference.getvrtdivspec(
        reference_inverse_gradient_u,
        reference_inverse_gradient_v,
        ntrunc=15,
    )
    reference_potential = np.squeeze(
        reference.spectogrd(
            spharm._spherepack.invlap(
                inverse_gradient_divergence,
                sg.EARTH_RADIUS_M,
            )
        )
    )
    reference_lap_vorticity = spharm._spherepack.lap(
        reference_vorticity,
        sg.EARTH_RADIUS_M,
    )
    reference_lap_divergence = spharm._spherepack.lap(
        reference_divergence,
        sg.EARTH_RADIUS_M,
    )
    reference_lap_u, reference_lap_v = reference.getuv(
        reference_lap_vorticity,
        reference_lap_divergence,
    )
    reference_lap_u = np.squeeze(reference_lap_u)
    reference_lap_v = np.squeeze(reference_lap_v)

    regridded = sg.regrid_vector(u, v, target, spectral="T7")
    decomposed = sg.helmholtz(u, v)
    potential = sg.inverse_gradient(
        _dataarray(inverse_gradient_u, grid, "eastward"),
        _dataarray(inverse_gradient_v, grid, "northward"),
    )
    laplacian = sg.vector_laplacian(u, v)
    inverse_laplacian = sg.inverse_vector_laplacian(
        _dataarray(_north_to_south(reference_lap_u), grid, "u"),
        _dataarray(_north_to_south(reference_lap_v), grid, "v"),
    )

    np.testing.assert_allclose(
        _north_to_south(np.asarray(regridded.u.values)),
        reference_target_u,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(regridded.v.values)),
        reference_target_v,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(decomposed.u_divergent.values)),
        reference_divergent_u,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(decomposed.v_divergent.values)),
        reference_divergent_v,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(decomposed.u_rotational.values)),
        reference_rotational_u,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(decomposed.v_rotational.values)),
        reference_rotational_v,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(potential.values)),
        reference_potential,
        rtol=0.0,
        atol=_INVERSE_GRADIENT_POTENTIAL_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(laplacian.u.values)),
        reference_lap_u,
        rtol=0.0,
        atol=_VECTOR_LAPLACIAN_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(laplacian.v.values)),
        reference_lap_v,
        rtol=0.0,
        atol=_VECTOR_LAPLACIAN_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(inverse_laplacian.u.values)),
        reference_u,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(inverse_laplacian.v.values)),
        reference_v,
        rtol=0.0,
        atol=_VECTOR_MAP_ATOL,
    )
