# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the JAX backend."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, tree_util

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid.grids import grids_equivalent
from tests.optional.jax._fields import scalar_values, vector_values

pytestmark = pytest.mark.jax_x64


def test_scalar_spectral_field_is_a_jit_compatible_pytree(
    gl_grid: sg.Grid,
) -> None:
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    spectral = sgj.analyze(field, grid=gl_grid)

    assert isinstance(spectral, sgj.SpectralField)
    assert len(tree_util.tree_leaves(spectral)) == 1
    np.testing.assert_allclose(spectral.synthesize(), field, atol=2.0e-12)
    np.testing.assert_allclose(
        spectral.laplacian().synthesize(),
        sgj.laplacian(field, grid=gl_grid),
        atol=2.0e-12,
    )
    compiled = jit(lambda value: value.laplacian().synthesize())
    np.testing.assert_allclose(compiled(spectral), spectral.laplacian().synthesize())


def test_jax_reusable_filters_skip_true_no_ops(gl_grid: sg.Grid) -> None:
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    scalar = sgj.analyze(field, grid=gl_grid)
    eastward_values, northward_values = vector_values(gl_grid)
    vector = sgj.analyze_vector(
        jnp.asarray(eastward_values, dtype=jnp.float64),
        jnp.asarray(northward_values, dtype=jnp.float64),
        grid=gl_grid,
    )

    assert scalar.filter() is scalar
    assert scalar.filter(scalar.spec) is scalar
    assert vector.filter() is vector
    assert vector.filter(vector.spec) is vector


def test_vector_spectral_field_matches_one_shot_operations_under_jit(
    cc_grid: sg.Grid,
) -> None:
    eastward_values, northward_values = vector_values(cc_grid)
    eastward = jnp.asarray(eastward_values, dtype=jnp.float64)
    northward = jnp.asarray(northward_values, dtype=jnp.float64)
    spectral = sgj.analyze_vector(eastward, northward, grid=cc_grid)

    assert isinstance(spectral, sgj.SpectralVectorField)
    vorticity = jit(lambda value: value.vorticity().synthesize())(spectral)
    np.testing.assert_allclose(
        vorticity,
        sgj.vorticity(eastward, northward, grid=cc_grid),
        atol=2.0e-12,
    )


def test_restricted_scalar_domain_is_preserved_on_jax_regrid(
    gl_grid: sg.Grid,
    cc_target_grid: sg.Grid,
) -> None:
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    spectral = sgj.analyze(field, "T4", grid=gl_grid)

    np.testing.assert_allclose(
        spectral.regrid(cc_target_grid),
        sgj.regrid(field, cc_target_grid, "T4", source_grid=gl_grid),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        spectral.regrid(cc_target_grid, taper=0.1),
        sgj.regrid(
            field,
            cc_target_grid,
            "T4",
            source_grid=gl_grid,
            taper=0.1,
        ),
        atol=2.0e-12,
    )
    compiled = jit(lambda value: value.regrid(cc_target_grid, taper=0.1))
    np.testing.assert_allclose(
        compiled(spectral),
        spectral.regrid(cc_target_grid, taper=0.1),
        atol=2.0e-12,
    )

    with pytest.raises(ValueError, match="exceeds"):
        spectral.regrid(cc_target_grid, "T5")


def test_unrestricted_domain_intersects_a_smaller_jax_target(
    gl_grid: sg.Grid,
    cc_target_grid: sg.Grid,
) -> None:
    field = jnp.asarray(scalar_values(gl_grid), dtype=jnp.float64)
    spectral = sgj.analyze(field, grid=gl_grid)

    np.testing.assert_allclose(
        spectral.regrid(cc_target_grid),
        sgj.regrid(field, cc_target_grid, source_grid=gl_grid),
        atol=2.0e-12,
    )


def test_restricted_vector_domain_is_preserved_on_jax_regrid(
    gl_grid: sg.Grid,
    cc_target_grid: sg.Grid,
) -> None:
    eastward_values, northward_values = vector_values(gl_grid)
    eastward = jnp.asarray(eastward_values, dtype=jnp.float64)
    northward = jnp.asarray(northward_values, dtype=jnp.float64)
    spectral = sgj.analyze_vector(eastward, northward, "T4", grid=gl_grid)

    result_u, result_v = spectral.regrid(cc_target_grid)
    expected_u, expected_v = sgj.regrid_vector(
        eastward,
        northward,
        cc_target_grid,
        "T4",
        source_grid=gl_grid,
    )
    np.testing.assert_allclose(result_u, expected_u, atol=2.0e-12)
    np.testing.assert_allclose(result_v, expected_v, atol=2.0e-12)

    result_u, result_v = spectral.regrid(cc_target_grid, taper=0.1)
    expected_u, expected_v = sgj.regrid_vector(
        eastward,
        northward,
        cc_target_grid,
        "T4",
        source_grid=gl_grid,
        taper=0.1,
    )
    np.testing.assert_allclose(result_u, expected_u, atol=2.0e-12)
    np.testing.assert_allclose(result_v, expected_v, atol=2.0e-12)


@pytest.mark.parametrize(
    ("kind", "latitude_order", "lon0", "ordering", "expected_order_kind"),
    [
        ("gl", "ascending", 0.0, "identity", "identity"),
        ("gl", "descending", -180.0, "identity", "roll"),
        ("cc", "ascending", 37.0, "identity", "roll"),
        ("cc", "descending", 0.0, "identity", "identity"),
        ("gl", "descending", 37.0, "roll", "roll"),
        ("cc", "ascending", -180.0, "roll", "roll"),
        ("gl", "ascending", 0.0, "arbitrary", "indices"),
        ("cc", "descending", 37.0, "arbitrary", "indices"),
    ],
)
def test_jax_pytree_grid_key_round_trip_preserves_geometry_and_order(
    kind: Literal["gl", "cc"],
    latitude_order: Literal["ascending", "descending"],
    lon0: float,
    ordering: str,
    expected_order_kind: str,
) -> None:
    from tests.optional.jax._fields import make_grid

    base = make_grid(
        kind,
        nlat=8 if kind == "gl" else 9,
        latitude_order=latitude_order,
        lon0=lon0,
    )
    if ordering == "roll":
        grid = sg.Grid(base.kind, base.latitude, np.roll(base.longitude, 3))
    elif ordering == "arbitrary":
        permutation = np.concatenate(
            (np.arange(0, base.nlon, 2), np.arange(1, base.nlon, 2))
        )
        grid = sg.Grid(base.kind, base.latitude, base.longitude[permutation])
    else:
        grid = base

    field = jnp.asarray(scalar_values(grid), dtype=jnp.float64)
    spectral = sgj.analyze(field, "T4", grid=grid)
    leaves, treedef = tree_util.tree_flatten(spectral)
    source_key = spectral.tree_flatten()[1][0][0]

    assert source_key[:4] == (kind, grid.nlat, grid.nlon, latitude_order)
    assert source_key[5][0] == expected_order_kind
    assert not any(isinstance(value, np.ndarray) for value in source_key)

    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    assert grids_equivalent(spectral.grid, reconstructed.grid)
    np.testing.assert_allclose(
        reconstructed.synthesize(), spectral.synthesize(), atol=2.0e-12
    )


def test_jax_noncanonical_pytree_grid_key_works_under_jit() -> None:
    from tests.optional.jax._fields import make_grid

    base = make_grid("cc", nlat=9, latitude_order="ascending", lon0=-180.0)
    grid = sg.Grid(base.kind, base.latitude, np.roll(base.longitude, 3))
    field = jnp.asarray(scalar_values(grid), dtype=jnp.float64)
    spectral = sgj.analyze(field, "T4", grid=grid)
    leaves, treedef = tree_util.tree_flatten(spectral)
    reconstructed = tree_util.tree_unflatten(treedef, leaves)

    compiled = jit(lambda value: value.synthesize())(reconstructed)
    np.testing.assert_allclose(compiled, spectral.synthesize(), atol=2.0e-12)


def test_jax_vector_pytree_grid_round_trip_preserves_axis_order(
    gl_grid: sg.Grid,
) -> None:
    grid = sg.Grid(gl_grid.kind, gl_grid.latitude, np.roll(gl_grid.longitude, 2))
    eastward_values, northward_values = vector_values(grid)
    eastward = jnp.asarray(eastward_values, dtype=jnp.float64)
    northward = jnp.asarray(northward_values, dtype=jnp.float64)
    spectral = sgj.analyze_vector(eastward, northward, "T4", grid=grid)

    leaves, treedef = tree_util.tree_flatten(spectral)
    reconstructed = tree_util.tree_unflatten(treedef, leaves)
    assert grids_equivalent(spectral.grid, reconstructed.grid)
    actual_u, actual_v = reconstructed.synthesize()
    expected_u, expected_v = spectral.synthesize()
    np.testing.assert_allclose(actual_u, expected_u, atol=2.0e-12)
    np.testing.assert_allclose(actual_v, expected_v, atol=2.0e-12)
