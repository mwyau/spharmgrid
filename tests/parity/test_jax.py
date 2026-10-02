# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical parity between the S2FFT/JAX and DUCC execution paths."""

from __future__ import annotations

import importlib
from typing import Literal

import numpy as np
import pytest

import spharmgrid as sg
from spharmgrid.grids import grid_capabilities
from spharmgrid.jax._backend import _make_transform
from spharmgrid.spectral import parse_spectral, resolve_transform_spec
from tests.optional.jax._fields import (
    as_xarray,
    make_grid,
    scalar_values,
    vector_values,
)

jax = pytest.importorskip("jax")
pytest.importorskip("s2fft")
jnp = importlib.import_module("jax.numpy")
config = jax.config
sgj = importlib.import_module("spharmgrid.jax")

pytestmark = pytest.mark.parity

GridKind = Literal["gl", "cc"]


def _tolerances(family: str) -> tuple[float, float]:
    """Return measured absolute-error floors with a small stability margin."""
    atols = {
        "scalar_map": 2.0e-13,  # measured max 9.4369e-14
        "vector_regrid": 3.0e-13,  # measured max 1.6432e-13
        "gradient": 1.0e-19,  # measured max 4.7778e-20
        "inverse_gradient": 1.0e-14,  # measured max 7.1055e-15
        "laplacian": 5.0e-26,  # measured max 3.7626e-26
        "inverse_laplacian": 5.0e-1,  # measured max 0.404296875
        "vector_laplacian": 2.0e-25,  # measured max 1.1082e-25
        "inverse_vector_laplacian": 1.5,  # measured max 1.03125
        "kinematics": 2.0e-19,  # measured max 1.1761e-19
        "potential": 3.0e-7,  # measured max 2.1095e-7
        "wind": 2.0e-13,  # measured max 1.1636e-13
        "helmholtz": 2.0e-13,  # measured max 1.3301e-13
    }
    return 0.0, atols[family]


def _assert_close(actual: object, expected: object, family: str) -> None:
    rtol, atol = _tolerances(family)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
    )


def _require_x64() -> None:
    if not config.read("jax_enable_x64"):
        pytest.skip("float64 parity requires JAX x64 enabled")


def _source_grid(kind: GridKind) -> sg.Grid:
    gl = kind == "gl"
    return make_grid(
        kind,
        nlat=8 if gl else 9,
        latitude_order="descending" if gl else "ascending",
        lon0=37.0 if gl else -75.0,
    )


def _target_grid(kind: GridKind) -> sg.Grid:
    gl = kind == "gl"
    return make_grid(
        kind,
        nlat=10 if gl else 11,
        latitude_order="ascending" if gl else "descending",
        lon0=-75.0 if gl else 123.0,
    )


def _as_jax(values: np.ndarray) -> jax.Array:
    return jnp.asarray(values, dtype=jnp.float64)


def test_jax_reusable_spectral_fields_match_ducc() -> None:
    _require_x64()
    grid = _source_grid("gl")
    scalar = scalar_values(grid).astype(np.float64)
    field = _as_jax(scalar)
    reference = as_xarray(scalar, grid)
    spectral = sgj.analyze(field, grid=grid)
    _assert_close(spectral.synthesize(), reference, "scalar_map")
    _assert_close(
        spectral.laplacian().synthesize(),
        sg.laplacian(reference),
        "laplacian",
    )

    eastward, northward = vector_values(grid)
    eastward = eastward.astype(np.float64)
    northward = northward.astype(np.float64)
    u = _as_jax(eastward)
    v = _as_jax(northward)
    reference_u = as_xarray(eastward, grid, "u")
    reference_v = as_xarray(northward, grid, "v")
    vector = sgj.analyze_vector(u, v, grid=grid)
    _assert_close(
        vector.vorticity().synthesize(),
        sg.vorticity(reference_u, reference_v),
        "kinematics",
    )
    _assert_close(
        vector.divergence().synthesize(),
        sg.divergence(reference_u, reference_v),
        "kinematics",
    )


@pytest.mark.parametrize("kind", ["gl", "cc"])
def test_scalar_jax_ducc_parity_and_scalar_regridding(
    kind: GridKind,
) -> None:
    _require_x64()
    source_grid = _source_grid(kind)
    scalar = scalar_values(source_grid).astype(np.float64)
    field = _as_jax(scalar)
    reference = as_xarray(scalar, source_grid)

    _assert_close(
        sgj.filter(field, "T3", grid=source_grid),
        sg.filter(reference, "T3"),
        "scalar_map",
    )
    _assert_close(
        sgj.filter(field, "T1-4", grid=source_grid),
        sg.filter(reference, "T1-4"),
        "scalar_map",
    )
    _assert_close(
        sgj.filter(field, "T4", taper=0.1, grid=source_grid),
        sg.filter(reference, "T4", taper=0.1),
        "scalar_map",
    )
    for notation in ("T4x2", "R2"):
        _assert_close(
            sgj.filter(field, notation, grid=source_grid),
            sg.filter(reference, notation),
            "scalar_map",
        )

    for target_kind in ("gl", "cc"):
        target_grid = _target_grid(target_kind)
        for notation in ("T4", "T4x2", "R2"):
            expected = sg.regrid(reference, target_grid, notation)
            actual = sgj.regrid(
                field,
                target_grid,
                notation,
                source_grid=source_grid,
            )
            _assert_close(actual, expected, "scalar_map")


@pytest.mark.parametrize("kind", ["gl", "cc"])
def test_all_vector_jax_ducc_operations_and_wind_sources(
    kind: GridKind,
) -> None:
    _require_x64()
    source_grid = _source_grid(kind)
    eastward, northward = vector_values(source_grid)
    eastward = eastward.astype(np.float64)
    northward = northward.astype(np.float64)
    u = _as_jax(eastward)
    v = _as_jax(northward)
    reference_u = as_xarray(eastward, source_grid, "u")
    reference_v = as_xarray(northward, source_grid, "v")

    for target_kind in ("gl", "cc"):
        target_grid = _target_grid(target_kind)
        for notation in ("T4", "T4x2", "R2"):
            expected = sg.regrid_vector(reference_u, reference_v, target_grid, notation)
            actual = sgj.regrid_vector(
                u,
                v,
                target_grid,
                notation,
                source_grid=source_grid,
            )
            _assert_close(actual[0], expected.u, "vector_regrid")
            _assert_close(actual[1], expected.v, "vector_regrid")

    expected_gradient = sg.gradient(as_xarray(scalar_values(source_grid), source_grid))
    scalar = _as_jax(scalar_values(source_grid).astype(np.float64))
    actual_gradient = sgj.gradient(scalar, grid=source_grid)
    _assert_close(actual_gradient[0], expected_gradient.gradient_eastward, "gradient")
    _assert_close(actual_gradient[1], expected_gradient.gradient_northward, "gradient")
    gradient_u = _as_jax(expected_gradient.gradient_eastward.values)
    gradient_v = _as_jax(expected_gradient.gradient_northward.values)
    expected_inverse_gradient = sg.inverse_gradient(
        expected_gradient.gradient_eastward,
        expected_gradient.gradient_northward,
    )
    _assert_close(
        sgj.inverse_gradient(gradient_u, gradient_v, grid=source_grid),
        expected_inverse_gradient,
        "inverse_gradient",
    )

    scalar_reference = as_xarray(scalar_values(source_grid), source_grid)
    _assert_close(
        sgj.laplacian(scalar, grid=source_grid),
        sg.laplacian(scalar_reference),
        "laplacian",
    )
    _assert_close(
        sgj.inverse_laplacian(scalar, grid=source_grid),
        sg.inverse_laplacian(scalar_reference),
        "inverse_laplacian",
    )

    expected_vector_laplacian = sg.vector_laplacian(reference_u, reference_v)
    actual_vector_laplacian = sgj.vector_laplacian(u, v, grid=source_grid)
    _assert_close(
        actual_vector_laplacian[0],
        expected_vector_laplacian.u,
        "vector_laplacian",
    )
    _assert_close(
        actual_vector_laplacian[1],
        expected_vector_laplacian.v,
        "vector_laplacian",
    )
    expected_inverse_vector_laplacian = sg.inverse_vector_laplacian(
        reference_u,
        reference_v,
    )
    actual_inverse_vector_laplacian = sgj.inverse_vector_laplacian(
        u,
        v,
        grid=source_grid,
    )
    _assert_close(
        actual_inverse_vector_laplacian[0],
        expected_inverse_vector_laplacian.u,
        "inverse_vector_laplacian",
    )
    _assert_close(
        actual_inverse_vector_laplacian[1],
        expected_inverse_vector_laplacian.v,
        "inverse_vector_laplacian",
    )

    expected_vorticity = sg.vorticity(reference_u, reference_v)
    expected_divergence = sg.divergence(reference_u, reference_v)
    actual_vorticity, actual_divergence = sgj.kinematics(u, v, grid=source_grid)
    _assert_close(actual_vorticity, expected_vorticity, "kinematics")
    _assert_close(actual_divergence, expected_divergence, "kinematics")
    _assert_close(
        sgj.vorticity(u, v, grid=source_grid),
        expected_vorticity,
        "kinematics",
    )
    _assert_close(
        sgj.divergence(u, v, grid=source_grid),
        expected_divergence,
        "kinematics",
    )

    expected_potentials = sg.potentials(reference_u, reference_v)
    actual_potentials = sgj.potentials(u, v, grid=source_grid)
    _assert_close(actual_potentials[0], expected_potentials.strf, "potential")
    _assert_close(actual_potentials[1], expected_potentials.vp, "potential")
    _assert_close(
        sgj.streamfunction(u, v, grid=source_grid),
        expected_potentials.strf,
        "potential",
    )
    _assert_close(
        sgj.velocity_potential(u, v, grid=source_grid),
        expected_potentials.vp,
        "potential",
    )

    expected_helmholtz = sg.helmholtz(reference_u, reference_v)
    actual_helmholtz = sgj.helmholtz(u, v, grid=source_grid)
    for actual, name in zip(
        actual_helmholtz,
        ("u_divergent", "v_divergent", "u_rotational", "v_rotational"),
        strict=True,
    ):
        _assert_close(actual, expected_helmholtz[name], "helmholtz")

    source_values = {
        "vorticity": expected_vorticity,
        "streamfunction": expected_potentials.strf,
        "divergence": expected_divergence,
        "velocity_potential": expected_potentials.vp,
    }
    for source in ("vorticity", "streamfunction"):
        actual = sgj.rotational_wind(
            _as_jax(source_values[source].values),
            grid=source_grid,
            source=source,
        )
        expected = sg.rotational_wind(source_values[source], source=source)
        _assert_close(actual[0], expected.u_rotational, "wind")
        _assert_close(actual[1], expected.v_rotational, "wind")
    for source in ("divergence", "velocity_potential"):
        actual = sgj.divergent_wind(
            _as_jax(source_values[source].values),
            grid=source_grid,
            source=source,
        )
        expected = sg.divergent_wind(source_values[source], source=source)
        _assert_close(actual[0], expected.u_divergent, "wind")
        _assert_close(actual[1], expected.v_divergent, "wind")
    for source, first_name, second_name in (
        ("vorticity_divergence", "vorticity", "divergence"),
        ("potentials", "streamfunction", "velocity_potential"),
    ):
        actual = sgj.wind(
            _as_jax(source_values[first_name].values),
            _as_jax(source_values[second_name].values),
            grid=source_grid,
            source=source,
        )
        expected = sg.wind(
            source_values[first_name],
            source_values[second_name],
            source=source,
        )
        _assert_close(actual[0], expected.u, "wind")
        _assert_close(actual[1], expected.v, "wind")


@pytest.mark.parametrize("nlon", [9, 16, 20])
def test_arbitrary_gl_scalar_vector_and_kinematics_match_ducc(nlon: int) -> None:
    _require_x64()
    grid = make_grid(
        "gl",
        nlat=8,
        nlon=nlon,
        latitude_order="descending",
        lon0=37.0,
    )
    scalar = scalar_values(grid).astype(np.float64)
    field = _as_jax(scalar)
    reference = as_xarray(scalar, grid)
    mmax = min(grid.nlat - 1, (nlon - 1) // 2)
    expected_spec = resolve_transform_spec(grid, grid, None)

    assert expected_spec.lmax == grid.nlat - 1
    assert expected_spec.mmax == mmax
    assert grid_capabilities(grid).longitude_mmax == (nlon - 1) // 2
    assert _make_transform(grid, grid, None).spec == expected_spec

    spectral = sgj.analyze(field, grid=grid)
    assert spectral.spec == expected_spec
    _assert_close(spectral.synthesize(), reference, "scalar_map")
    for notation in ("T4", "T6x4", "R2"):
        _assert_close(
            sgj.filter(field, notation, grid=grid),
            sg.filter(reference, notation),
            "scalar_map",
        )

    eastward, northward = vector_values(grid)
    u = _as_jax(eastward.astype(np.float64))
    v = _as_jax(northward.astype(np.float64))
    reference_u = as_xarray(eastward, grid, "u")
    reference_v = as_xarray(northward, grid, "v")
    actual_vector = sgj.regrid_vector(u, v, grid, source_grid=grid)
    expected_vector = sg.regrid_vector(reference_u, reference_v, grid)
    _assert_close(actual_vector[0], expected_vector.u, "vector_regrid")
    _assert_close(actual_vector[1], expected_vector.v, "vector_regrid")

    actual_vorticity, actual_divergence = sgj.kinematics(u, v, grid=grid)
    expected = sg.kinematics(reference_u, reference_v)
    _assert_close(actual_vorticity, expected["vo"], "kinematics")
    _assert_close(actual_divergence, expected["d"], "kinematics")


@pytest.mark.parametrize(
    ("source_nlon", "target_nlon"),
    [(9, 20), (20, 9), (15, 16)],
)
def test_arbitrary_gl_regridding_intersects_grid_capabilities(
    source_nlon: int, target_nlon: int
) -> None:
    _require_x64()
    source = make_grid(
        "gl", nlat=8, nlon=source_nlon, latitude_order="descending", lon0=37.0
    )
    target = make_grid(
        "gl", nlat=8, nlon=target_nlon, latitude_order="ascending", lon0=-75.0
    )
    source_spec = resolve_transform_spec(source, target, None)
    transform = _make_transform(source, target, None)
    assert transform.spec == source_spec
    assert source_spec.mmax == min(
        source.nlat - 1,
        (source.nlon - 1) // 2,
        (target.nlon - 1) // 2,
    )

    scalar = scalar_values(source).astype(np.float64)
    field = _as_jax(scalar)
    reference = as_xarray(scalar, source)
    _assert_close(
        sgj.regrid(field, target, source_grid=source),
        sg.regrid(reference, target),
        "scalar_map",
    )

    eastward, northward = vector_values(source)
    u = _as_jax(eastward.astype(np.float64))
    v = _as_jax(northward.astype(np.float64))
    expected = sg.regrid_vector(
        as_xarray(eastward, source, "u"),
        as_xarray(northward, source, "v"),
        target,
    )
    actual = sgj.regrid_vector(u, v, target, source_grid=source)
    _assert_close(actual[0], expected.u, "vector_regrid")
    _assert_close(actual[1], expected.v, "vector_regrid")

    for notation in ("T4", "T6x4", "T4x2", "R2"):
        selection = parse_spectral(notation)
        selected_spec = resolve_transform_spec(source, target, selection)
        assert _make_transform(source, target, selection).spec == selected_spec
        _assert_close(
            sgj.regrid(field, target, notation, source_grid=source),
            sg.regrid(reference, target, notation),
            "scalar_map",
        )
        expected_selected_vector = sg.regrid_vector(
            as_xarray(eastward, source, "u"),
            as_xarray(northward, source, "v"),
            target,
            notation,
        )
        actual_selected_vector = sgj.regrid_vector(
            u, v, target, notation, source_grid=source
        )
        _assert_close(
            actual_selected_vector[0], expected_selected_vector.u, "vector_regrid"
        )
        _assert_close(
            actual_selected_vector[1], expected_selected_vector.v, "vector_regrid"
        )


def test_underresolved_gl_rejects_unavailable_explicit_bandwidth() -> None:
    _require_x64()
    grid = make_grid("gl", nlat=8, nlon=9)
    field = _as_jax(scalar_values(grid).astype(np.float64))
    with pytest.raises(ValueError, match="supported triangular bandwidth 4"):
        sgj.filter(field, "T5", grid=grid)
