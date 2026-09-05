"""Optional Gaussian-grid parity checks against pyspharm-syl/SPHEREPACK."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pytest

import spharmgrid as sg
from tests.conftest import scalar_field, solid_body_wind

pytestmark = pytest.mark.parity
spharm = pytest.importorskip(
    "spharm",
    reason="run with uv --group parity on a supported Python version",
)


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
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_eastward.values)),
        reference_east,
        rtol=3.0e-6,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_northward.values)),
        reference_north,
        rtol=3.0e-6,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(regridded.values)),
        reference_regridded,
        rtol=3.0e-6,
        atol=3.0e-6,
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

    # pyspharm-syl's SPHEREPACK wrapper returns regular-grid scalar maps at
    # float32 precision, so this comparison uses its documented ~1e-6 scale.
    np.testing.assert_allclose(
        _north_to_south(np.asarray(filtered.values)),
        reference_filtered,
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_eastward.values)),
        reference_east,
        rtol=3.0e-6,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(gradient.gradient_northward.values)),
        reference_north,
        rtol=3.0e-6,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(regridded.values)),
        reference_regridded,
        rtol=3.0e-6,
        atol=3.0e-6,
    )


def test_gaussian_wind_kinematics_potentials_and_inverse_match_pyspharm() -> None:
    grid = _gaussian_grid()
    u, v = solid_body_wind(grid)
    # Include a degree-one divergent component so SPHEREPACK's potential
    # comparison has a resolved signal rather than testing only its
    # float32-level zero-divergence residual.
    v = v + (7.0 * np.cos(np.deg2rad(grid.latitude))[:, None] * np.ones((1, grid.nlon)))
    reference = _reference_transform()
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
        rtol=3.0e-6,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(diagnostics.d.values)),
        reference_d,
        rtol=3.0e-6,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(potential.strf.values)),
        reference_psi,
        rtol=3.0e-6,
        atol=3.0e-4,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(potential.vp.values)),
        reference_chi,
        rtol=3.0e-6,
        atol=3.0e-4,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(reconstructed.u.values)),
        reference_u_reconstructed,
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    np.testing.assert_allclose(
        _north_to_south(np.asarray(reconstructed.v.values)),
        reference_v_reconstructed,
        rtol=3.0e-6,
        atol=3.0e-6,
    )
