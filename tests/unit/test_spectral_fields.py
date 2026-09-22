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
from spharmgrid.grids import grids_equivalent
from tests.conftest import scalar_field, solid_body_wind, supported_grid


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
