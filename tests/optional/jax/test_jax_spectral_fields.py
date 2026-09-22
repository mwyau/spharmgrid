# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable analyzed representations for the JAX backend."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit, tree_util

import spharmgrid as sg
import spharmgrid.jax as sgj
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


def test_jax_pytree_grid_key_is_compact_and_exact_for_cyclic_order(
    gl_grid: sg.Grid,
) -> None:
    rolled = sg.Grid(gl_grid.kind, gl_grid.latitude, np.roll(gl_grid.longitude, 2))
    field = jnp.asarray(scalar_values(rolled), dtype=jnp.float64)
    spectral = sgj.analyze(field, "T4", grid=rolled)

    leaves, aux_data = spectral.tree_flatten()
    key = aux_data[0][0]
    assert key[:4] == ("gl", rolled.nlat, rolled.nlon, "descending")
    assert key[5][0] == "roll"
    assert not any(isinstance(value, np.ndarray) for value in key)

    restored = tree_util.tree_unflatten(tree_util.tree_structure(spectral), leaves)
    np.testing.assert_allclose(restored.synthesize(), spectral.synthesize(), atol=0.0)
