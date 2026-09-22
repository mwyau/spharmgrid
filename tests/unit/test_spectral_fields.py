# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the DUCC/Xarray API."""

from __future__ import annotations

from importlib import import_module
from typing import Literal
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
import spharmgrid._spectral_field as spectral_field_module
from spharmgrid._ducc import (
    alm_degrees,
    alm_orders,
    geometry_for,
    scalar_synthesis,
)
from spharmgrid._transform import TransformSpec
from spharmgrid.grids import grid_layout, grids_equivalent
from spharmgrid.metadata import (
    operator_metadata,
    output_metadata,
    vector_operator_metadata,
    wind_component_metadata,
)
from tests.conftest import scalar_field, solid_body_wind, supported_grid


def _scalar_mode(
    grid: sg.Grid,
    degree: int,
    order: int,
    *,
    spec: TransformSpec,
) -> xr.DataArray:
    """Return one packed scalar harmonic on a supplied transform domain."""
    degrees = alm_degrees(spec.lmax, spec.mmax)
    index = np.flatnonzero(
        (degrees == degree) & (alm_orders(spec.lmax, spec.mmax) == order)
    )
    assert index.size == 1
    coefficients = np.zeros((1, degrees.size), dtype=np.complex128)
    coefficients[0, index[0]] = 1.0
    layout = grid_layout(grid)
    values = scalar_synthesis(
        coefficients,
        spec=spec,
        geometry=geometry_for(grid),
        ntheta=grid.nlat,
        nphi=grid.nlon,
        phi0=layout.phi0_radians,
        nthreads=0,
    )
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
    )


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_scalar_analysis_synthesis_preserves_layout_and_metadata(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind, latitude_order="descending", lon0=-180.0)
    field = scalar_field(grid, leading=True, name="air_temperature")
    field.attrs["standard_name"] = "air_temperature"

    spectral = sg.analyze(field)
    reconstructed = spectral.synthesize()

    assert isinstance(spectral, sg.SpectralField)
    assert grids_equivalent(spectral.grid, grid)
    assert reconstructed.dims == field.dims
    xr.testing.assert_identical(reconstructed.lat, field.lat)
    xr.testing.assert_identical(reconstructed.lon, field.lon)
    np.testing.assert_allclose(reconstructed, field, rtol=0.0, atol=2.0e-14)
    assert reconstructed.name == field.name
    assert reconstructed.attrs == field.attrs


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_scalar_spectral_methods_match_one_shot_operations(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    target = supported_grid("gl" if kind == "cc" else "cc")
    field = scalar_field(grid)
    spectral = sg.analyze(field)

    np.testing.assert_allclose(
        spectral.filter("T2").synthesize(),
        sg.filter(field, "T2"),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        spectral.filter("T2", taper=0.1).synthesize(),
        sg.filter(field, "T2", taper=0.1),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        spectral.laplacian().synthesize(),
        sg.laplacian(field),
        rtol=0.0,
        atol=2.0e-25,
    )
    np.testing.assert_allclose(
        spectral.inverse_laplacian().synthesize(),
        sg.inverse_laplacian(field),
        rtol=0.0,
        atol=2.0e-2,
    )
    np.testing.assert_allclose(
        spectral.regrid(target),
        sg.regrid(field, target),
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_reusable_filter_skips_true_no_op(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    scalar = sg.analyze(scalar_field(grid))
    u, v = solid_body_wind(grid)
    vector = sg.analyze_vector(u, v)

    assert scalar.filter() is scalar
    assert scalar.filter(scalar.spec) is scalar
    assert vector.filter() is vector
    assert vector.filter(vector.spec) is vector


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_vector_spectral_methods_match_one_shot_operations(
    kind: Literal["cc", "gl"],
) -> None:
    grid = supported_grid(kind)
    target = supported_grid("gl" if kind == "cc" else "cc")
    u, v = solid_body_wind(grid)
    spectral = sg.analyze_vector(u, v)

    assert isinstance(spectral, sg.SpectralVectorField)
    u_reconstructed, v_reconstructed = spectral.synthesize()
    np.testing.assert_allclose(u_reconstructed, u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(v_reconstructed, v, rtol=0.0, atol=3.0e-14)

    vector_laplacian = spectral.laplacian().synthesize()
    expected_laplacian = sg.vector_laplacian(u, v)
    np.testing.assert_allclose(
        vector_laplacian[0], expected_laplacian.u, rtol=0.0, atol=2.0e-25
    )
    np.testing.assert_allclose(
        vector_laplacian[1], expected_laplacian.v, rtol=0.0, atol=2.0e-25
    )

    inverse_vector_laplacian = spectral.inverse_laplacian().synthesize()
    expected_inverse = sg.inverse_vector_laplacian(u, v)
    np.testing.assert_allclose(
        inverse_vector_laplacian[0],
        expected_inverse.u,
        rtol=0.0,
        atol=2.0e-2,
    )
    np.testing.assert_allclose(
        inverse_vector_laplacian[1],
        expected_inverse.v,
        rtol=0.0,
        atol=2.0e-2,
    )

    np.testing.assert_allclose(
        spectral.vorticity().synthesize(),
        sg.vorticity(u, v),
        rtol=0.0,
        atol=2.0e-19,
    )
    np.testing.assert_allclose(
        spectral.divergence().synthesize(),
        sg.divergence(u, v),
        rtol=0.0,
        atol=2.0e-19,
    )
    np.testing.assert_allclose(
        spectral.streamfunction().synthesize(),
        sg.streamfunction(u, v),
        rtol=0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        spectral.velocity_potential().synthesize(),
        sg.velocity_potential(u, v),
        rtol=0.0,
        atol=2.0e-7,
    )

    filtered = spectral.filter("T2").synthesize()
    expected_filtered = sg.regrid_vector(u, v, grid, "T2")
    np.testing.assert_allclose(filtered[0], expected_filtered.u, atol=3.0e-14)
    np.testing.assert_allclose(filtered[1], expected_filtered.v, atol=3.0e-14)

    divergent = spectral.divergent().synthesize()
    rotational = spectral.rotational().synthesize()
    expected = sg.helmholtz(u, v)
    np.testing.assert_allclose(divergent[0], expected.u_divergent, atol=3.0e-14)
    np.testing.assert_allclose(divergent[1], expected.v_divergent, atol=3.0e-14)
    np.testing.assert_allclose(rotational[0], expected.u_rotational, atol=3.0e-14)
    np.testing.assert_allclose(rotational[1], expected.v_rotational, atol=3.0e-14)

    regridded = spectral.regrid(target)
    expected_regridded = sg.regrid_vector(u, v, target)
    np.testing.assert_allclose(regridded[0], expected_regridded.u, atol=3.0e-14)
    np.testing.assert_allclose(regridded[1], expected_regridded.v, atol=3.0e-14)


def test_compound_xarray_operations_share_one_vector_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    kinematics_module = import_module("spharmgrid.kinematics")
    analysis = Mock(wraps=kinematics_module.vector_analysis)
    monkeypatch.setattr(kinematics_module, "vector_analysis", analysis)

    sg.kinematics(u, v)
    assert analysis.call_count == 1
    analysis.reset_mock()
    sg.potentials(u, v)
    assert analysis.call_count == 1
    analysis.reset_mock()
    sg.helmholtz(u, v)
    assert analysis.call_count == 1


def test_reusable_vector_methods_do_not_reanalyze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    analysis = Mock(wraps=spectral_field_module.vector_analysis)
    monkeypatch.setattr(spectral_field_module, "vector_analysis", analysis)

    spectral = sg.analyze_vector(u, v)
    spectral.vorticity().synthesize()
    spectral.divergence().synthesize()
    spectral.streamfunction().synthesize()

    assert analysis.call_count == 1


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_restricted_scalar_domain_is_preserved_on_synthesis_and_regrid(
    kind: Literal["cc", "gl"],
) -> None:
    source = supported_grid(kind)
    target = supported_grid("gl" if kind == "cc" else "cc")
    field = scalar_field(source)
    spectral = sg.analyze(field, "T2")

    np.testing.assert_allclose(
        spectral.synthesize(),
        sg.filter(field, "T2"),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        spectral.regrid(target),
        sg.regrid(field, target, "T2"),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        spectral.regrid(target, taper=0.1),
        sg.regrid(field, target, "T2", taper=0.1),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        spectral.regrid(target, "T1"),
        sg.regrid(field, target, "T1"),
        rtol=0.0,
        atol=2.0e-14,
    )


def test_unrestricted_domain_intersects_a_smaller_target_grid() -> None:
    source = supported_grid("cc")
    target = sg.gaussian_grid(6, 12)
    field = scalar_field(source)
    spectral = sg.analyze(field)

    np.testing.assert_allclose(
        spectral.regrid(target),
        sg.regrid(field, target),
        rtol=0.0,
        atol=3.0e-14,
    )

    u, v = solid_body_wind(source)
    vector = sg.analyze_vector(u, v)
    result_u, result_v = vector.regrid(target)
    expected = sg.regrid_vector(u, v, target)
    np.testing.assert_allclose(result_u, expected.u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(result_v, expected.v, rtol=0.0, atol=3.0e-14)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_restricted_vector_domain_is_preserved_on_regrid(
    kind: Literal["cc", "gl"],
) -> None:
    source = supported_grid(kind)
    target = supported_grid("gl" if kind == "cc" else "cc")
    u, v = solid_body_wind(source)
    spectral = sg.analyze_vector(u, v, "T2")

    result = spectral.regrid(target)
    expected = sg.regrid_vector(u, v, target, "T2")
    np.testing.assert_allclose(result[0], expected.u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(result[1], expected.v, rtol=0.0, atol=3.0e-14)

    tapered = spectral.regrid(target, taper=0.1)
    expected_tapered = sg.regrid_vector(u, v, target, "T2", taper=0.1)
    np.testing.assert_allclose(tapered[0], expected_tapered.u, rtol=0.0, atol=3.0e-14)
    np.testing.assert_allclose(tapered[1], expected_tapered.v, rtol=0.0, atol=3.0e-14)


@pytest.mark.parametrize(
    ("notation", "modes", "narrower"),
    [
        ("T1-3", ((1, 0), (3, 1)), "T1-2"),
        ("T4x2", ((4, 2), (3, 3)), "T3x2"),
        ("R2", ((4, 2), (3, 0)), "R1"),
    ],
)
def test_reusable_shaped_domains_match_one_shot_operations(
    notation: str,
    modes: tuple[tuple[int, int], tuple[int, int]],
    narrower: str,
) -> None:
    source = sg.clenshaw_curtis_grid(8, 12, latitude_order="descending")
    target = sg.gaussian_grid(6, 12, latitude_order="descending")
    full = TransformSpec(0, 4, 4, "triangular")
    first_mode, second_mode = modes
    field = _scalar_mode(source, *first_mode, spec=full) + _scalar_mode(
        source, *second_mode, spec=full
    )
    spectral = sg.analyze(field, notation)

    np.testing.assert_allclose(
        spectral.synthesize(),
        sg.filter(field, notation),
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        spectral.regrid(target),
        sg.regrid(field, target, notation),
        rtol=0.0,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        spectral.filter(narrower).synthesize(),
        sg.filter(field, narrower),
        rtol=0.0,
        atol=3.0e-13,
    )


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_reusable_domain_cannot_be_expanded(
    kind: Literal["cc", "gl"],
) -> None:
    source = supported_grid(kind)
    target = supported_grid("gl" if kind == "cc" else "cc")
    field = scalar_field(source)
    u, v = solid_body_wind(source)

    with pytest.raises(ValueError, match="exceeds"):
        sg.analyze(field, "T2").filter("T3")
    with pytest.raises(ValueError, match="exceeds"):
        sg.analyze(field, "T2").regrid(target, "T3")
    with pytest.raises(ValueError, match="exceeds"):
        sg.analyze_vector(u, v, "T2").filter("T3")
    with pytest.raises(ValueError, match="exceeds"):
        sg.analyze_vector(u, v, "T2").regrid(target, "T3")


def test_reusable_scalar_operator_metadata_matches_one_shot_helpers() -> None:
    field = scalar_field(supported_grid("cc"), name="temperature")
    field.attrs["standard_name"] = "air_temperature"

    for operation in ("laplacian", "inverse_laplacian"):
        result = getattr(sg.analyze(field), operation)().synthesize()
        assert result.name == field.name
        assert result.attrs == operator_metadata(field, operation)


def test_reusable_vector_operator_metadata_matches_one_shot_helpers() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    u = u.copy(deep=False)
    v = v.copy(deep=False)
    u.name = "custom_u"
    v.name = "custom_v"
    u.attrs = {
        "standard_name": "eastward_wind",
        "long_name": "Custom eastward wind",
        "units": "m s-1",
    }
    v.attrs = {
        "standard_name": "northward_wind",
        "long_name": "Custom northward wind",
        "units": "m s-1",
    }
    spectral = sg.analyze_vector(u, v)

    for operation in ("laplacian", "inverse_laplacian"):
        result_u, result_v = getattr(spectral, operation)().synthesize()
        assert result_u.name == u.name
        assert result_v.name == v.name
        assert result_u.attrs == vector_operator_metadata(u, "eastward", operation)
        assert result_v.attrs == vector_operator_metadata(v, "northward", operation)
        assert "standard_name" not in result_u.attrs
        assert "standard_name" not in result_v.attrs


def test_reusable_wind_component_metadata_survives_chaining() -> None:
    source = supported_grid("cc")
    target = supported_grid("gl")
    u, v = solid_body_wind(source)
    spectral = sg.analyze_vector(u, v)

    divergent_u, divergent_v = spectral.divergent().filter("T2").regrid(target)
    assert divergent_u.name == "u_divergent"
    assert divergent_v.name == "v_divergent"
    assert divergent_u.attrs == wind_component_metadata("eastward", "divergent")
    assert divergent_v.attrs == wind_component_metadata("northward", "divergent")

    rotational_u, rotational_v = spectral.rotational().filter("T2").regrid(target)
    assert rotational_u.name == "u_rotational"
    assert rotational_v.name == "v_rotational"
    assert rotational_u.attrs == wind_component_metadata("eastward", "rotational")
    assert rotational_v.attrs == wind_component_metadata("northward", "rotational")


@pytest.mark.parametrize(
    ("method", "quantity"),
    [
        ("vorticity", "vo"),
        ("divergence", "d"),
        ("streamfunction", "strf"),
        ("velocity_potential", "vp"),
    ],
)
def test_reusable_scalar_diagnostic_metadata_is_canonical(
    method: Literal["vorticity", "divergence", "streamfunction", "velocity_potential"],
    quantity: Literal["vo", "d", "strf", "vp"],
) -> None:
    source = supported_grid("cc")
    target = supported_grid("gl")
    u, v = solid_body_wind(source)
    result = getattr(sg.analyze_vector(u, v), method)().filter("T2").regrid(target)

    assert result.name == quantity
    assert result.attrs == output_metadata(quantity)
