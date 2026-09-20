# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental xarray containers over the tensor-native JAX API."""

from __future__ import annotations

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from jax import Array, grad, jit, make_jaxpr

pytest.importorskip("xarray_jax")

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid.metadata import find_variable
from tests.optional.jax._fields import scalar_values, vector_values


def _coords(field: xr.DataArray, grid: sg.Grid) -> dict[str, object]:
    """Build static output coordinates for a field on ``grid``."""
    coordinates: dict[str, object] = {
        str(field.dims[-2]): grid.latitude,
        str(field.dims[-1]): grid.longitude,
    }
    for dimension in field.dims[:-2]:
        if dimension in field.coords:
            coordinates[str(dimension)] = field.coords[dimension].data
    return coordinates


def _field(
    values: np.ndarray,
    grid: sg.Grid,
    *,
    leading_dim: str | None = None,
    leading_coord: np.ndarray | None = None,
    name: str = "field",
    attrs: dict[str, str] | None = None,
) -> xr.DataArray:
    """Wrap an analytic field in a standard xarray object with JAX data."""
    if leading_dim is None:
        dims = ("latitude", "longitude")
        coordinates: dict[str, object] = {
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        }
    else:
        if leading_coord is None:
            raise ValueError("leading_coord is required with leading_dim")
        dims = (leading_dim, "latitude", "longitude")
        coordinates = {
            leading_dim: leading_coord,
            "latitude": grid.latitude,
            "longitude": grid.longitude,
        }
    return xr.DataArray(
        jnp.asarray(values, dtype=jnp.float64),
        dims=dims,
        coords=coordinates,
        name=name,
        attrs=attrs,
    )


def _wrap(
    data: Array,
    template: xr.DataArray,
    grid: sg.Grid,
    *,
    name: str | None = None,
    attrs: dict[str, str] | None = None,
) -> xr.DataArray:
    """Construct a labeled output without touching the numerical data."""
    return xr.DataArray(
        data,
        dims=template.dims,
        coords=_coords(template, grid),
        name=template.name if name is None else name,
        attrs=template.attrs if attrs is None else attrs,
    )


def _filter_dataarray(field: xr.DataArray, grid: sg.Grid) -> xr.DataArray:
    """Apply the raw JAX filter and restore the input labels."""
    data = sgj.filter(field.data, "T4", grid=grid)
    return _wrap(data, field, grid)


def _regrid_dataarray(
    field: xr.DataArray,
    source_grid: sg.Grid,
    target_grid: sg.Grid,
) -> xr.DataArray:
    """Apply the raw JAX regrid and construct target-grid coordinates."""
    data = sgj.regrid(
        field.data,
        target_grid,
        "T4",
        source_grid=source_grid,
    )
    return _wrap(data, field, target_grid)


def _kinematics_dataarrays(
    u: xr.DataArray,
    v: xr.DataArray,
    grid: sg.Grid,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply the raw JAX vector operation and label both outputs."""
    vorticity, divergence = sgj.kinematics(u.data, v.data, grid=grid)
    return (
        _wrap(
            vorticity,
            u,
            grid,
            name="vo",
            attrs={"standard_name": "atmosphere_relative_vorticity"},
        ),
        _wrap(
            divergence,
            v,
            grid,
            name="d",
            attrs={"standard_name": "divergence_of_wind"},
        ),
    )


def _dataset_kinematics(dataset: xr.Dataset, grid: sg.Grid) -> xr.Dataset:
    """Apply the raw JAX vector operation to two Dataset variables."""
    vorticity, divergence = sgj.kinematics(
        dataset["u"].data,
        dataset["v"].data,
        grid=grid,
    )
    coordinates = _coords(dataset["u"], grid)
    return xr.Dataset(
        {
            "vo": (
                dataset["u"].dims,
                vorticity,
                {"standard_name": "atmosphere_relative_vorticity"},
            ),
            "d": (
                dataset["v"].dims,
                divergence,
                {"standard_name": "divergence_of_wind"},
            ),
        },
        coords=coordinates,
        attrs=dataset.attrs,
    )


def _assert_jax_float64(field: xr.DataArray) -> None:
    assert isinstance(field.data, Array)
    assert field.data.dtype == jnp.float64


def _assert_horizontal_coordinates(field: xr.DataArray, grid: sg.Grid) -> None:
    np.testing.assert_array_equal(field.coords[field.dims[-2]], grid.latitude)
    np.testing.assert_array_equal(field.coords[field.dims[-1]], grid.longitude)


def test_xarray_jax_registration_is_explicit() -> None:
    """Registration is opt-in, and importing the package enables the PyTree."""
    script = """
import jax
import jax.numpy as jnp
import xarray as xr

field = xr.DataArray(jnp.ones((2,)), dims=('x',))
try:
    jax.jit(lambda value: value)(field)
except TypeError:
    pass
else:
    raise AssertionError('xarray unexpectedly registered before opt-in import')

import xarray_jax

result = jax.jit(lambda value: value)(field)
assert isinstance(result.data, jax.Array)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "JAX_ENABLE_X64": "1"},
    )


def test_dataarray_filter_jit_and_grad_match_raw_jax_api(
    gl_grid: sg.Grid,
) -> None:
    atol = 2.0e-13
    values = scalar_values(gl_grid)
    field = _field(
        values,
        gl_grid,
        name="temperature",
        attrs={"units": "K", "long_name": "test field"},
    )
    expected = sgj.filter(field.data, "T4", grid=gl_grid)

    eager = _filter_dataarray(field, gl_grid)
    actual = jit(lambda value: _filter_dataarray(value, gl_grid))(field)
    _assert_jax_float64(actual)
    assert actual.dims == field.dims
    assert actual.name == field.name
    _assert_horizontal_coordinates(actual, gl_grid)
    np.testing.assert_allclose(
        np.asarray(eager.data), np.asarray(expected), rtol=0.0, atol=atol
    )
    np.testing.assert_allclose(
        np.asarray(actual.data), np.asarray(expected), rtol=0.0, atol=atol
    )

    def loss(value: xr.DataArray) -> Array:
        return jnp.sum(_filter_dataarray(value, gl_grid).data ** 2)

    gradient = jit(grad(loss))(field)
    expected_gradient = grad(
        lambda value: jnp.sum(sgj.filter(value, "T4", grid=gl_grid) ** 2)
    )(field.data)
    _assert_jax_float64(gradient)
    assert gradient.dims == field.dims
    assert gradient.name == field.name
    _assert_horizontal_coordinates(gradient, gl_grid)
    assert bool(jnp.isfinite(gradient.data).all())
    np.testing.assert_allclose(
        np.asarray(gradient.data),
        np.asarray(expected_gradient),
        rtol=0.0,
        atol=atol,
    )


def test_static_coordinates_retrace_and_leading_dimensions_survive(
    gl_grid: sg.Grid,
) -> None:
    atol = 2.0e-13
    values = np.stack((scalar_values(gl_grid), 2.0 * scalar_values(gl_grid)))
    field = _field(
        values,
        gl_grid,
        leading_dim="time",
        leading_coord=np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]"),
    )
    expected = sgj.filter(field.data, "T4", grid=gl_grid)

    actual = jit(lambda value: _filter_dataarray(value, gl_grid))(field)
    _assert_jax_float64(actual)
    assert actual.dims == ("time", "latitude", "longitude")
    assert actual.shape == field.shape
    np.testing.assert_array_equal(actual.coords["time"], field.coords["time"])
    _assert_horizontal_coordinates(actual, gl_grid)
    np.testing.assert_allclose(
        np.asarray(actual.data), np.asarray(expected), rtol=0.0, atol=atol
    )

    def identity(value: xr.DataArray) -> xr.DataArray:
        assert isinstance(value.coords["latitude"].data, np.ndarray)
        return value

    compiled_identity = jit(identity)
    original = compiled_identity(field)
    repeated = compiled_identity(field)
    changed_coordinates = field.assign_coords(
        latitude=field.coords["latitude"].data + 0.25
    )
    changed = compiled_identity(changed_coordinates)
    np.testing.assert_array_equal(original.coords["latitude"], field.coords["latitude"])
    np.testing.assert_array_equal(repeated.coords["latitude"], field.coords["latitude"])
    np.testing.assert_array_equal(
        changed.coords["latitude"], changed_coordinates.coords["latitude"]
    )


def test_regrid_jit_builds_new_target_coordinates(
    gl_grid: sg.Grid,
    gl_target_grid: sg.Grid,
) -> None:
    atol = 2.0e-13
    field = _field(scalar_values(gl_grid), gl_grid, name="field")
    expected = sgj.regrid(
        field.data,
        gl_target_grid,
        "T4",
        source_grid=gl_grid,
    )
    actual = jit(lambda value: _regrid_dataarray(value, gl_grid, gl_target_grid))(field)

    _assert_jax_float64(actual)
    assert actual.dims == field.dims
    assert actual.shape == (gl_target_grid.nlat, gl_target_grid.nlon)
    assert actual.name == field.name
    _assert_horizontal_coordinates(actual, gl_target_grid)
    np.testing.assert_allclose(
        np.asarray(actual.data), np.asarray(expected), rtol=0.0, atol=atol
    )


def test_vector_kinematics_jit_and_grad_match_raw_jax_api(
    cc_grid: sg.Grid,
) -> None:
    atol = 2.0e-19
    u_values, v_values = vector_values(cc_grid)
    u = _field(u_values, cc_grid, name="u", attrs={"units": "m s-1"})
    v = _field(v_values, cc_grid, name="v", attrs={"units": "m s-1"})
    expected_vorticity, expected_divergence = sgj.kinematics(
        u.data, v.data, grid=cc_grid
    )

    eager_vorticity, eager_divergence = _kinematics_dataarrays(u, v, cc_grid)
    actual_vorticity, actual_divergence = jit(
        lambda first, second: _kinematics_dataarrays(first, second, cc_grid)
    )(u, v)
    for outputs in (
        (eager_vorticity, eager_divergence),
        (actual_vorticity, actual_divergence),
    ):
        for actual in outputs:
            _assert_jax_float64(actual)
            assert actual.dims == u.dims
            _assert_horizontal_coordinates(actual, cc_grid)
        assert outputs[0].name == "vo"
        assert outputs[1].name == "d"
        np.testing.assert_allclose(
            np.asarray(outputs[0].data),
            np.asarray(expected_vorticity),
            rtol=0.0,
            atol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(outputs[1].data),
            np.asarray(expected_divergence),
            rtol=0.0,
            atol=atol,
        )

    def loss(first: xr.DataArray) -> Array:
        vorticity, _ = _kinematics_dataarrays(first, v, cc_grid)
        return jnp.sum(vorticity.data**2)

    gradient = jit(grad(loss))(u)
    expected_gradient = grad(
        lambda first: jnp.sum(sgj.kinematics(first, v.data, grid=cc_grid)[0] ** 2)
    )(u.data)
    _assert_jax_float64(gradient)
    assert gradient.dims == u.dims
    assert gradient.name == u.name
    _assert_horizontal_coordinates(gradient, cc_grid)
    assert bool(jnp.isfinite(gradient.data).all())
    np.testing.assert_allclose(
        np.asarray(gradient.data),
        np.asarray(expected_gradient),
        rtol=0.0,
        atol=atol,
    )


def test_dataset_cf_discovery_does_not_survive_jax_boundary() -> None:
    """Characterize loss of attrs needed by the CF-aware Dataset API."""
    data = jnp.arange(4, dtype=jnp.float64).reshape(2, 2)
    dataset = xr.Dataset(
        {
            "eastward": xr.DataArray(
                data,
                dims=("latitude", "longitude"),
                attrs={"standard_name": "eastward_wind", "units": "m s-1"},
            ),
            "northward": xr.DataArray(
                data + 1.0,
                dims=("latitude", "longitude"),
                attrs={"standard_name": "northward_wind", "units": "m s-1"},
            ),
        },
        coords={
            "latitude": np.array([-90.0, 90.0]),
            "longitude": np.array([0.0, 180.0]),
        },
        attrs={"source": "experiment"},
    )

    assert find_variable(dataset, "u").name == "eastward"
    assert find_variable(dataset, "v").name == "northward"

    round_tripped = jit(lambda value: value)(dataset)
    assert round_tripped.attrs == {}
    assert round_tripped["eastward"].attrs == {}
    assert round_tripped["northward"].attrs == {}

    with pytest.raises(ValueError, match="could not identify 'u'"):
        find_variable(round_tripped, "u")
    with pytest.raises(ValueError, match="could not identify 'v'"):
        find_variable(round_tripped, "v")

    # Noncanonical names make the failure visible instead of letting the
    # canonical ``u``/``v`` fallback hide missing CF attrs.  Dataset ``.sg``
    # wind methods use this resolver, so xarray_jax carries numerical data,
    # dimensions, names, and static coordinates through JAX transforms, but
    # ordinary attrs are not PyTree metadata.  Future ``.sg`` JAX integration
    # must resolve CF semantics and restore metadata outside the compiled
    # boundary; ``spharmgrid.jax.*`` remains the pure JAX-array layer.


def test_dataset_kinematics_is_jittable_with_labeled_outputs(
    cc_grid: sg.Grid,
) -> None:
    atol = 2.0e-19
    u_values, v_values = vector_values(cc_grid)
    u = _field(u_values, cc_grid, name="u")
    v = _field(v_values, cc_grid, name="v")
    dataset = xr.Dataset({"u": u, "v": v}, attrs={"source": "experiment"})

    actual = jit(lambda value: _dataset_kinematics(value, cc_grid))(dataset)
    expected_vorticity, expected_divergence = sgj.kinematics(
        u.data, v.data, grid=cc_grid
    )

    assert set(actual.data_vars) == {"vo", "d"}
    assert dict(actual.sizes) == {
        "latitude": cc_grid.nlat,
        "longitude": cc_grid.nlon,
    }
    _assert_jax_float64(actual["vo"])
    _assert_jax_float64(actual["d"])
    _assert_horizontal_coordinates(actual["vo"], cc_grid)
    np.testing.assert_allclose(
        np.asarray(actual["vo"].data),
        np.asarray(expected_vorticity),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual["d"].data),
        np.asarray(expected_divergence),
        rtol=0.0,
        atol=atol,
    )


def test_dataarray_adapter_jaxpr_has_no_host_callbacks(
    gl_grid: sg.Grid,
) -> None:
    field = _field(scalar_values(gl_grid), gl_grid)
    jaxpr = str(make_jaxpr(lambda value: _filter_dataarray(value, gl_grid))(field))
    for callback in ("pure_callback", "io_callback", "device_get"):
        assert callback not in jaxpr
