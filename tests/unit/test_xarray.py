"""xarray leading-dimension, metadata, and Dask behavior tests."""

from __future__ import annotations

import cftime
import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import scalar_field, solid_body_wind, supported_grid


def test_leading_dimensions_and_time_coordinates_are_preserved() -> None:
    field = scalar_field(supported_grid("cc"), leading=True).expand_dims(
        time=np.array([np.datetime64("2000-01-01"), np.datetime64("2000-01-02")])
    )

    result = sg.filter(field, "T3")

    assert result.dims == field.dims
    xr.testing.assert_identical(result.member, field.member)
    xr.testing.assert_identical(result.time, field.time)
    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-14)


def test_detected_horizontal_dimensions_need_not_be_last_or_adjacent() -> None:
    field = scalar_field(supported_grid("cc"), leading=True).transpose(
        "lat", "member", "lon"
    )

    result = sg.filter(field, "T3")

    assert result.dims == field.dims
    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-14)


def test_cftime_calendar_coordinate_is_preserved() -> None:
    field = scalar_field(supported_grid("cc")).expand_dims(
        time=np.array(
            [
                cftime.DatetimeNoLeap(2001, 2, 28),
                cftime.DatetimeNoLeap(2001, 3, 1),
            ],
            dtype=object,
        )
    )

    result = sg.filter(field, "T3")

    assert list(result.time.values) == list(field.time.values)
    assert isinstance(result.time.values[0], cftime.DatetimeNoLeap)


def test_dask_input_stays_lazy_with_rechunked_horizontal_core_dimensions() -> None:
    field = scalar_field(supported_grid("cc"), leading=True).chunk(
        {"member": 1, "lat": 8, "lon": 12}
    )

    result = sg.filter(field, "T3")

    assert hasattr(result.data, "dask")
    np.testing.assert_allclose(
        result.compute(), field.compute(), rtol=0.0, atol=2.0e-14
    )


def test_mixed_eager_and_dask_wind_inputs_stay_lazy() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    dask_v = v.chunk({"lat": 8, "lon": 12})

    result = sg.kinematics(u, dask_v)

    assert hasattr(result.vo.data, "dask")
    expected = sg.kinematics(u, v)
    xr.testing.assert_allclose(result.compute(), expected, rtol=0.0, atol=0.0)


def test_new_vector_operations_keep_dask_inputs_lazy() -> None:
    grid = supported_grid("cc")
    target = supported_grid("gl")
    eager_u, eager_v = solid_body_wind(grid)
    member = xr.DataArray(["first", "second"], dims="member")
    eager_u = xr.concat([eager_u, 2.0 * eager_u], dim=member)
    eager_v = xr.concat([eager_v, 2.0 * eager_v], dim=member)
    u = eager_u.chunk({"member": 1, "lat": 8, "lon": 12})
    v = eager_v.chunk({"member": 1, "lat": 8, "lon": 12})

    regridded = sg.regrid_vector(u, v, target)
    decomposed = sg.helmholtz(u, v)
    laplacian = sg.vector_laplacian(u, v)
    restored = sg.inverse_vector_laplacian(laplacian.u, laplacian.v)
    scalar = scalar_field(grid, leading=True)
    eager_gradient = sg.gradient(scalar)
    dask_gradient = sg.gradient(scalar.chunk({"member": 1, "lat": 8, "lon": 12}))
    inverse_gradient = sg.inverse_gradient(
        dask_gradient.gradient_eastward,
        dask_gradient.gradient_northward,
    )

    assert all(
        hasattr(variable.data, "dask") for variable in regridded.data_vars.values()
    )
    assert all(
        hasattr(variable.data, "dask") for variable in decomposed.data_vars.values()
    )
    assert all(
        hasattr(variable.data, "dask") for variable in laplacian.data_vars.values()
    )
    assert all(
        hasattr(variable.data, "dask") for variable in restored.data_vars.values()
    )
    assert hasattr(inverse_gradient.data, "dask")
    xr.testing.assert_allclose(
        regridded.compute(),
        sg.regrid_vector(eager_u, eager_v, target),
        rtol=0.0,
        atol=0.0,
    )
    xr.testing.assert_allclose(
        decomposed.compute(),
        sg.helmholtz(eager_u, eager_v),
        rtol=0.0,
        atol=0.0,
    )
    eager_laplacian = sg.vector_laplacian(eager_u, eager_v)
    xr.testing.assert_allclose(
        laplacian.compute(), eager_laplacian, rtol=0.0, atol=0.0
    )
    xr.testing.assert_allclose(
        restored.compute(),
        sg.inverse_vector_laplacian(eager_laplacian.u, eager_laplacian.v),
        rtol=0.0,
        atol=0.0,
    )
    xr.testing.assert_allclose(
        inverse_gradient.compute(),
        sg.inverse_gradient(
            eager_gradient.gradient_eastward,
            eager_gradient.gradient_northward,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_filter_and_regrid_preserve_data_variable_semantics() -> None:
    field = scalar_field(supported_grid("cc"), name="air_temperature")
    field.attrs.update(
        {
            "standard_name": "air_temperature",
            "units": "K",
            "custom": "preserve me",
        }
    )

    filtered = sg.filter(field, "T3")
    regridded = sg.regrid(field, supported_grid("gl"))

    assert filtered.name == field.name
    assert regridded.name == field.name
    assert filtered.attrs == field.attrs
    assert regridded.attrs == field.attrs


def test_xarray_target_uses_reference_coordinates_and_dimension_names() -> None:
    source = scalar_field(supported_grid("cc"))
    target_grid = supported_grid("gl")
    reference = xr.DataArray(
        np.zeros((target_grid.nlat, target_grid.nlon)),
        dims=("latitude", "longitude"),
        coords={"latitude": target_grid.latitude, "longitude": target_grid.longitude},
    )

    result = sg.regrid(source, reference)

    assert result.dims == ("latitude", "longitude")
    np.testing.assert_array_equal(result.latitude, target_grid.latitude)
    np.testing.assert_array_equal(result.longitude, target_grid.longitude)


def test_incompatible_wind_coordinates_raise_before_transform() -> None:
    grid = supported_grid("cc")
    u = scalar_field(grid, name="u")
    v = scalar_field(grid, name="v").assign_coords(lon=grid.longitude + 1.0)

    with pytest.raises(ValueError, match="equivalent supported grids"):
        sg.vorticity(u, v)


def test_equivalent_wind_longitude_conventions_align_with_their_data() -> None:
    grid = supported_grid("cc", lon0=-180.0)
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    streamfunction = xr.DataArray(
        sg.EARTH_RADIUS_M * np.cos(latitude) * np.cos(longitude),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="strf",
    )
    velocity_potential = xr.DataArray(
        sg.EARTH_RADIUS_M * np.cos(latitude) * np.sin(longitude),
        dims=("lat", "lon"),
        coords={"lat": grid.latitude, "lon": grid.longitude},
        name="vp",
    )
    components = sg.wind(streamfunction, velocity_potential)
    v_zero_to_360 = components.v.assign_coords(lon=np.mod(components.lon, 360.0))
    v_zero_to_360 = v_zero_to_360.sortby("lon")

    expected = sg.kinematics(components.u, components.v)
    aligned = sg.kinematics(components.u, v_zero_to_360)

    xr.testing.assert_identical(aligned, expected)
