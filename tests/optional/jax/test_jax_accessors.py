# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the labeled Xarray layer over the JAX API."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from jax import Array, config, devices

import spharmgrid as sg
import spharmgrid.jax as sgj
from spharmgrid._accessors import (
    DataArrayAccessor as RootDataArrayAccessor,
)
from spharmgrid._accessors import DatasetAccessor as RootDatasetAccessor
from spharmgrid.jax._accessors import (
    DataArrayAccessor as JaxDataArrayAccessor,
)
from spharmgrid.jax._accessors import DatasetAccessor as JaxDatasetAccessor
from tests.optional.jax._fields import scalar_values, vector_values

pytestmark = pytest.mark.jax_x64


def _field(
    values: np.ndarray,
    grid: sg.Grid,
    *,
    name: str,
    attrs: dict[str, str] | None = None,
    leading: bool = False,
) -> xr.DataArray:
    if leading:
        data = np.stack((values, 2.0 * values), axis=1)
        dims = ("lat", "time", "lon")
        coords: dict[str, object] = {
            "lat": grid.latitude,
            "time": np.array(["first", "second"], dtype=object),
            "lon": grid.longitude,
        }
    else:
        data = values
        dims = ("lat", "lon")
        coords = {"lat": grid.latitude, "lon": grid.longitude}
    return xr.DataArray(
        jnp.asarray(data, dtype=jnp.float64),
        dims=dims,
        coords=coords,
        name=name,
        attrs=attrs,
    )


def _assert_raw_equal(
    actual: xr.DataArray,
    expected: Array,
    *,
    atol: float = 2.0e-12,
) -> None:
    assert isinstance(actual.data, Array)
    np.testing.assert_allclose(
        np.asarray(actual.data),
        np.asarray(expected),
        rtol=0.0,
        atol=atol,
    )


def test_registration_is_private_to_jax_and_does_not_need_xarray_jax() -> None:
    script = """
import builtins
import os
import sys

import spharmgrid
import xarray as xr

assert 'jax' not in sys.modules
assert 's2fft' not in sys.modules
assert not hasattr(xr.DataArray([1], dims=('x',)), 'sgj')

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'xarray_jax' or name.startswith('xarray_jax.'):
        raise ModuleNotFoundError('xarray_jax', name='xarray_jax')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import

import jax.numpy as jnp
import spharmgrid.jax as sgj

assert 'xarray_jax' not in sys.modules
field = xr.DataArray(
    jnp.ones((5, 8), dtype=jnp.float64),
    dims=('lat', 'lon'),
    coords={
        'lat': [-90.0, -45.0, 0.0, 45.0, 90.0],
        'lon': [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
    },
)
assert hasattr(field, 'sgj')
result = field.sgj.filter('T2')
assert isinstance(result.data, __import__('jax').Array)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "JAX_ENABLE_X64": "1"},
    )


def test_device_helpers_preserve_containers_metadata_and_coordinates(
    cc_grid: sg.Grid,
) -> None:
    values = scalar_values(cc_grid)
    field = _field(
        values,
        cc_grid,
        name="temperature",
        attrs={"units": "K", "long_name": "temperature"},
        leading=True,
    )
    dataset = xr.Dataset(
        {
            "u": field.rename("u"),
            "v": (0.5 * field).rename("v"),
        },
        attrs={"source": "test"},
    )
    before_x64 = config.read("jax_enable_x64")

    placed_field = sgj.device_put(field)
    placed_dataset = sgj.device_put(dataset, device=devices()[0])

    assert isinstance(placed_field, xr.DataArray)
    assert isinstance(placed_field.data, Array)
    assert placed_field.dims == field.dims
    assert placed_field.name == field.name
    assert placed_field.attrs == field.attrs
    assert np.asarray(placed_field.coords["lat"]).dtype == np.dtype("float64")
    assert not isinstance(placed_field.coords["lat"].data, Array)

    assert isinstance(placed_dataset, xr.Dataset)
    assert placed_dataset.attrs == dataset.attrs
    for name in dataset.data_vars:
        assert isinstance(placed_dataset[name].data, Array)
        assert placed_dataset[name].dims == dataset[name].dims
        assert placed_dataset[name].attrs == dataset[name].attrs
    np.testing.assert_array_equal(placed_dataset.coords["time"], dataset.coords["time"])
    assert not isinstance(placed_dataset.coords["time"].data, Array)
    assert config.read("jax_enable_x64") == before_x64

    host_field = sgj.device_get(placed_field)
    host_dataset = sgj.device_get(placed_dataset)
    assert isinstance(host_field.data, np.ndarray)
    assert isinstance(host_dataset["u"].data, np.ndarray)
    xr.testing.assert_identical(host_field, field)
    xr.testing.assert_identical(host_dataset, dataset)
    assert config.read("jax_enable_x64") == before_x64


def test_device_put_rejects_unsupported_payload() -> None:
    for values in (
        np.array([{"not": "numeric"}], dtype=object),
        np.array(["not numeric"]),
    ):
        field = xr.DataArray(values, dims=("x",), name="unsupported")
        with pytest.raises(TypeError, match="cannot be represented by JAX"):
            sgj.device_put(field)

    dataset = xr.Dataset({"unsupported": field})
    with pytest.raises(TypeError, match="Dataset data variable"):
        sgj.device_put(dataset)


def test_device_put_preserves_invalid_placement_error(cc_grid: sg.Grid) -> None:
    field = xr.DataArray(
        np.ones((cc_grid.nlat, cc_grid.nlon), dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": cc_grid.latitude, "lon": cc_grid.longitude},
    )
    with pytest.raises(ValueError, match="device_put") as error:
        sgj.device_put(field, device="not-a-device")
    assert "cannot be represented by JAX" not in str(error.value)


def test_sgj_preserves_non_string_dimension_names(cc_grid: sg.Grid) -> None:
    field = xr.DataArray(
        jnp.asarray(scalar_values(cc_grid)[None, :, :], dtype=jnp.float64),
        dims=(0, "lat", "lon"),
        coords={
            "time": xr.Variable((0,), np.array(["first"], dtype=object)),
            "lat": cc_grid.latitude,
            "lon": cc_grid.longitude,
        },
        name="field",
    )

    result = field.sgj.filter("T2")

    assert result.dims == (0, "lat", "lon")
    assert result.coords["time"].dims == (0,)
    np.testing.assert_array_equal(result.coords["time"], field.coords["time"])


def test_sgj_preserves_non_string_coordinate_names_without_collisions(
    cc_grid: sg.Grid,
) -> None:
    field = xr.DataArray(
        jnp.asarray(scalar_values(cc_grid)[None, :, :], dtype=jnp.float64),
        dims=("time", "lat", "lon"),
        coords={
            0: xr.Variable(("time",), np.array([1])),
            "0": xr.Variable(("time",), np.array([2])),
            "lat": cc_grid.latitude,
            "lon": cc_grid.longitude,
        },
        name="field",
    )

    result = field.sgj.filter("T2")

    assert 0 in result.coords
    assert "0" in result.coords
    assert result.coords[0].dims == ("time",)
    assert result.coords["0"].dims == ("time",)
    np.testing.assert_array_equal(result.coords[0], field.coords[0])
    np.testing.assert_array_equal(result.coords["0"], field.coords["0"])


def test_sgj_requires_explicit_jax_payload_placement(cc_grid: sg.Grid) -> None:
    values = scalar_values(cc_grid)
    numpy_field = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": cc_grid.latitude, "lon": cc_grid.longitude},
    )
    with pytest.raises(
        TypeError,
        match=r"\.sgj requires JAX-backed data; use spharmgrid\.jax\.device_put",
    ):
        numpy_field.sgj.filter("T2")

    jax_field = sgj.device_put(numpy_field)
    with pytest.raises(
        TypeError,
        match=r"\.sgj requires JAX-backed data; use spharmgrid\.jax\.device_put",
    ):
        jax_field.sgj.kinematics(numpy_field)


def test_dataarray_accessors_delegate_and_preserve_labels(
    gl_grid: sg.Grid,
    gl_target_grid: sg.Grid,
) -> None:
    values = scalar_values(gl_grid)
    u = _field(
        values,
        gl_grid,
        name="u",
        attrs={"units": "m s-1", "long_name": "eastward input"},
        leading=True,
    )
    v = _field(
        0.5 * values,
        gl_grid,
        name="v",
        attrs={"units": "m s-1", "long_name": "northward input"},
        leading=True,
    )
    raw_u = u.transpose("time", "lat", "lon").data
    raw_v = v.transpose("time", "lat", "lon").data

    filtered = u.sgj.filter("T4")
    _assert_raw_equal(
        filtered.transpose("time", "lat", "lon"), sgj.filter(raw_u, "T4", grid=gl_grid)
    )
    assert filtered.dims == u.dims
    assert filtered.name == u.name
    assert filtered.attrs == u.attrs
    np.testing.assert_array_equal(filtered.coords["time"], u.coords["time"])

    regridded = u.sgj.regrid(gl_target_grid, "T4")
    _assert_raw_equal(
        regridded.transpose("time", "lat", "lon"),
        sgj.regrid(raw_u, gl_target_grid, "T4", source_grid=gl_grid),
    )
    assert regridded.dims == u.dims
    assert regridded.sizes["lat"] == gl_target_grid.nlat
    assert regridded.sizes["lon"] == gl_target_grid.nlon
    np.testing.assert_array_equal(regridded.lat, gl_target_grid.latitude)
    np.testing.assert_array_equal(regridded.lon, gl_target_grid.longitude)

    regridded_vector = u.sgj.regrid_vector(v, gl_target_grid, "T4")
    raw_vector = sgj.regrid_vector(
        raw_u,
        raw_v,
        gl_target_grid,
        "T4",
        source_grid=gl_grid,
    )
    _assert_raw_equal(
        regridded_vector.u.transpose("time", "lat", "lon"),
        raw_vector[0],
    )
    _assert_raw_equal(
        regridded_vector.v.transpose("time", "lat", "lon"),
        raw_vector[1],
    )
    assert regridded_vector.u.attrs == u.attrs
    assert regridded_vector.v.attrs == v.attrs

    gradient = u.sgj.gradient()
    raw_gradient = sgj.gradient(raw_u, grid=gl_grid)
    _assert_raw_equal(
        gradient.gradient_eastward.transpose("time", "lat", "lon"),
        raw_gradient[0],
    )
    _assert_raw_equal(
        gradient.gradient_northward.transpose("time", "lat", "lon"),
        raw_gradient[1],
    )

    laplacian = u.sgj.laplacian()
    _assert_raw_equal(
        laplacian.transpose("time", "lat", "lon"),
        sgj.laplacian(raw_u, grid=gl_grid),
    )
    assert laplacian.attrs["long_name"].startswith("Laplacian of")

    kinematics = u.sgj.kinematics(v)
    raw_kinematics = sgj.kinematics(raw_u, raw_v, grid=gl_grid)
    _assert_raw_equal(kinematics.vo.transpose("time", "lat", "lon"), raw_kinematics[0])
    _assert_raw_equal(kinematics.d.transpose("time", "lat", "lon"), raw_kinematics[1])
    assert kinematics.vo.attrs["standard_name"] == "atmosphere_relative_vorticity"
    assert kinematics.d.attrs["standard_name"] == "divergence_of_wind"

    potentials = u.sgj.potentials(v)
    raw_potentials = sgj.potentials(raw_u, raw_v, grid=gl_grid)
    _assert_raw_equal(
        potentials.strf.transpose("time", "lat", "lon"), raw_potentials[0]
    )
    _assert_raw_equal(potentials.vp.transpose("time", "lat", "lon"), raw_potentials[1])

    helmholtz = u.sgj.helmholtz(v)
    raw_helmholtz = sgj.helmholtz(raw_u, raw_v, grid=gl_grid)
    for name, expected in zip(helmholtz.data_vars, raw_helmholtz, strict=True):
        _assert_raw_equal(helmholtz[name].transpose("time", "lat", "lon"), expected)

    reconstructed = kinematics.vo.sgj.wind(
        kinematics.d,
        source="vorticity_divergence",
    )
    raw_reconstructed = sgj.wind(
        raw_kinematics[0],
        raw_kinematics[1],
        grid=gl_grid,
        source="vorticity_divergence",
    )
    _assert_raw_equal(
        reconstructed.u.transpose("time", "lat", "lon"),
        raw_reconstructed[0],
    )
    _assert_raw_equal(
        reconstructed.v.transpose("time", "lat", "lon"),
        raw_reconstructed[1],
    )


def test_paired_accessor_preserves_cyclic_longitude_equivalence() -> None:
    first_grid = sg.clenshaw_curtis_grid(9, 16, lon0=-180.0)
    second_grid = sg.clenshaw_curtis_grid(9, 16, lon0=0.0)
    latitude = np.deg2rad(first_grid.latitude)[:, None]
    first_longitude = np.deg2rad(first_grid.longitude)[None, :]
    second_longitude = np.deg2rad(second_grid.longitude)[None, :]
    u_values = np.broadcast_to(
        np.cos(latitude) * np.cos(first_longitude),
        (first_grid.nlat, first_grid.nlon),
    ).copy()
    v_values = np.broadcast_to(
        np.sin(latitude) * np.sin(second_longitude),
        (second_grid.nlat, second_grid.nlon),
    ).copy()
    u = _field(u_values, first_grid, name="u")
    v = _field(v_values, second_grid, name="v")

    actual = u.sgj.kinematics(v)
    expected = sgj.kinematics(
        u.data,
        jnp.asarray(
            np.broadcast_to(
                np.sin(latitude) * np.sin(first_longitude),
                (first_grid.nlat, first_grid.nlon),
            ).copy()
        ),
        grid=first_grid,
    )
    _assert_raw_equal(actual.vo, expected[0])
    _assert_raw_equal(actual.d, expected[1])
    np.testing.assert_array_equal(actual.lat, first_grid.latitude)
    np.testing.assert_array_equal(actual.lon, first_grid.longitude)


def test_paired_accessor_uses_first_dimension_order(
    gl_grid: sg.Grid,
    gl_target_grid: sg.Grid,
) -> None:
    values = scalar_values(gl_grid)
    time = np.array(["first", "second"], dtype=object)
    u = xr.DataArray(
        jnp.asarray(np.stack((values, 2.0 * values), axis=0), dtype=jnp.float64),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": gl_grid.latitude, "lon": gl_grid.longitude},
        name="u",
        attrs={"units": "m s-1", "long_name": "eastward input"},
    )
    v = xr.DataArray(
        jnp.asarray(
            np.stack((0.5 * values, 1.5 * values), axis=0).transpose(1, 0, 2),
            dtype=jnp.float64,
        ),
        dims=("lat", "time", "lon"),
        coords={"lat": gl_grid.latitude, "time": time, "lon": gl_grid.longitude},
        name="v",
        attrs={"units": "m s-1", "long_name": "northward input"},
    )
    raw_u = u.transpose("time", "lat", "lon").data
    raw_v = v.transpose("time", "lat", "lon").data

    actual_kinematics = u.sgj.kinematics(v)
    expected_kinematics = sgj.kinematics(raw_u, raw_v, grid=gl_grid)
    for name, expected in zip(("vo", "d"), expected_kinematics, strict=True):
        assert actual_kinematics[name].dims == ("time", "lat", "lon")
        _assert_raw_equal(actual_kinematics[name], expected)

    actual_regrid = u.sgj.regrid_vector(v, gl_target_grid, "T4")
    expected_regrid = sgj.regrid_vector(
        raw_u,
        raw_v,
        gl_target_grid,
        "T4",
        source_grid=gl_grid,
    )
    for name, expected in zip(("u", "v"), expected_regrid, strict=True):
        assert actual_regrid[name].dims == ("time", "lat", "lon")
        _assert_raw_equal(actual_regrid[name].transpose("time", "lat", "lon"), expected)


def _method_signature(
    accessor: type[object], name: str, *, excluded: set[str]
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    return tuple(
        (
            parameter.name,
            parameter.kind,
            parameter.default,
        )
        for parameter in inspect.signature(getattr(accessor, name)).parameters.values()
        if parameter.name not in excluded
    )


def _public_methods(accessor: type[object]) -> set[str]:
    return {
        name
        for name, value in vars(accessor).items()
        if not name.startswith("_") and callable(value)
    }


def test_sg_and_sgj_accessor_signatures_match() -> None:
    exceptions = {"sht_threads"}
    jax_only_root_methods = {"analyze", "analyze_vector"}
    dataarray_methods = _public_methods(RootDataArrayAccessor)
    dataset_methods = _public_methods(RootDatasetAccessor)

    assert dataarray_methods - jax_only_root_methods == _public_methods(
        JaxDataArrayAccessor
    )
    assert dataset_methods - jax_only_root_methods == _public_methods(
        JaxDatasetAccessor
    )
    for name in dataarray_methods - jax_only_root_methods:
        assert _method_signature(
            RootDataArrayAccessor,
            name,
            excluded=exceptions,
        ) == _method_signature(JaxDataArrayAccessor, name, excluded=set())
    for name in dataset_methods - jax_only_root_methods:
        assert _method_signature(
            RootDatasetAccessor,
            name,
            excluded=exceptions,
        ) == _method_signature(JaxDatasetAccessor, name, excluded=set())


def test_dataset_accessor_discovers_cf_variables(cc_grid: sg.Grid) -> None:
    u_values, v_values = vector_values(cc_grid)
    u = _field(
        u_values,
        cc_grid,
        name="eastward_input",
        attrs={"standard_name": "eastward_wind", "units": "m s-1"},
    )
    v = _field(
        v_values,
        cc_grid,
        name="northward_input",
        attrs={"standard_name": "northward_wind", "units": "m s-1"},
    )
    dataset = sgj.device_put(
        xr.Dataset(
            {"eastward_input": u, "northward_input": v},
            attrs={"source": "test"},
        )
    )
    result = dataset.sgj.kinematics(
        vorticity="relative_vorticity",
        divergence="wind_divergence",
    )
    expected = sgj.kinematics(u.data, v.data, grid=cc_grid)

    assert dataset.attrs == {"source": "test"}
    assert tuple(result.data_vars) == ("relative_vorticity", "wind_divergence")
    _assert_raw_equal(result.relative_vorticity, expected[0])
    _assert_raw_equal(result.wind_divergence, expected[1])
    assert (
        result.relative_vorticity.attrs["standard_name"]
        == "atmosphere_relative_vorticity"
    )
    assert result.wind_divergence.attrs["standard_name"] == "divergence_of_wind"


def test_dataset_accessor_preserves_discovery_errors(cc_grid: sg.Grid) -> None:
    u_values, v_values = vector_values(cc_grid)
    u = _field(
        u_values,
        cc_grid,
        name="u",
        attrs={"standard_name": "eastward_wind"},
    )
    v = _field(
        v_values,
        cc_grid,
        name="v",
        attrs={"standard_name": "northward_wind"},
    )

    with pytest.raises(ValueError, match="could not identify 'v'"):
        xr.Dataset({"u": u}).sgj.kinematics()

    duplicate = u.rename("duplicate_u")
    with pytest.raises(ValueError, match="ambiguous 'u'"):
        xr.Dataset({"u": u, "duplicate_u": duplicate, "v": v}).sgj.vorticity()

    vo = _field(
        scalar_values(cc_grid),
        cc_grid,
        name="vo",
        attrs={"standard_name": "atmosphere_relative_vorticity"},
    )
    streamfunction = _field(
        scalar_values(cc_grid),
        cc_grid,
        name="strf",
        attrs={"standard_name": "atmosphere_horizontal_streamfunction"},
    )
    with pytest.raises(ValueError, match="both eligible scalar sources"):
        xr.Dataset({"vo": vo, "strf": streamfunction}).sgj.rotational_wind()


def test_sgj_accessor_surface_is_explicit() -> None:
    dataarray_methods = {
        "filter",
        "regrid",
        "regrid_vector",
        "gradient",
        "laplacian",
        "inverse_laplacian",
        "inverse_gradient",
        "vorticity",
        "divergence",
        "kinematics",
        "streamfunction",
        "velocity_potential",
        "potentials",
        "helmholtz",
        "vector_laplacian",
        "inverse_vector_laplacian",
        "rotational_wind",
        "divergent_wind",
        "wind",
    }
    dataset_methods = {
        "regrid_vector",
        "inverse_gradient",
        "vorticity",
        "divergence",
        "kinematics",
        "streamfunction",
        "velocity_potential",
        "potentials",
        "helmholtz",
        "vector_laplacian",
        "inverse_vector_laplacian",
        "rotational_wind",
        "divergent_wind",
        "wind",
    }
    field_accessor = xr.DataArray([1], dims=("x",)).sgj
    dataset_accessor = xr.Dataset().sgj
    assert dataarray_methods <= set(dir(field_accessor))
    assert dataset_methods <= set(dir(dataset_accessor))
    assert not hasattr(field_accessor, "analyze")
    assert not hasattr(field_accessor, "analyze_vector")
    assert not hasattr(dataset_accessor, "analyze")
    assert not hasattr(dataset_accessor, "analyze_vector")
