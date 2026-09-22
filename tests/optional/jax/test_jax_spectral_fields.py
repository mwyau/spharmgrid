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
