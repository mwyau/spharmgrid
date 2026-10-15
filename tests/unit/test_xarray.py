# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""xarray leading-dimension, metadata, and Dask behavior tests."""

from __future__ import annotations

import cftime
import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import scalar_field, supported_grid


def test_leading_dimensions_and_time_coordinates_are_preserved() -> None:
    field = scalar_field(supported_grid("cc"), leading=True).expand_dims(
        time=np.array([np.datetime64("2000-01-01"), np.datetime64("2000-01-02")])
    )

    result = sg.filter(field, "T3")

    assert result.dims == field.dims
    xr.testing.assert_identical(result.member, field.member)
    xr.testing.assert_identical(result.time, field.time)
    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-14)


def test_multiple_leading_dimensions_and_interleaved_spatial_axes_are_preserved() -> (
    None
):
    grid = supported_grid("cc")
    target = supported_grid("gl")
    base = scalar_field(grid)
    scales = np.arange(1.0, 7.0).reshape(2, 3)
    values = scales[..., None, None] * np.asarray(base)
    field = xr.DataArray(
        values,
        dims=("member", "level", "lat", "lon"),
        coords={
            "member": ["first", "second"],
            "level": [1000, 850, 500],
            "lat": grid.latitude,
            "lon": grid.longitude,
        },
        name=base.name,
        attrs=base.attrs,
    ).transpose("lat", "member", "lon", "level")

    filtered = sg.filter(field, "T3")
    regridded = sg.regrid(field, target, "T3")
    reconstructed = sg.analyze(field).filter("T3").synthesize()

    for result in (filtered, regridded, reconstructed):
        assert result.dims == field.dims
        xr.testing.assert_identical(result.member, field.member)
        xr.testing.assert_identical(result.level, field.level)
        assert result.name == field.name
        assert result.attrs == field.attrs
    xr.testing.assert_allclose(filtered, field, rtol=0.0, atol=3.0e-14)
    xr.testing.assert_allclose(reconstructed, filtered, rtol=0.0, atol=3.0e-14)

    expected_regridded = xr.concat(
        [
            xr.concat(
                [
                    sg.regrid(base * scales[member, level], target, "T3")
                    for level in range(scales.shape[1])
                ],
                dim=xr.DataArray([1000, 850, 500], dims="level", name="level"),
            )
            for member in range(scales.shape[0])
        ],
        dim=xr.DataArray(["first", "second"], dims="member", name="member"),
    ).transpose(*field.dims)
    xr.testing.assert_allclose(regridded, expected_regridded, rtol=0.0, atol=0.0)


def test_batched_vector_operations_match_independent_fields() -> None:
    grid = supported_grid("cc")
    target = supported_grid("gl")
    latitude = np.deg2rad(grid.latitude)[:, None]
    longitude = np.deg2rad(grid.longitude)[None, :]
    coordinates = {"lat": grid.latitude, "lon": grid.longitude}
    streamfunction = xr.DataArray(
        3.0 * sg.EARTH_RADIUS_M * np.cos(latitude) * np.cos(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
        name="strf",
    )
    velocity_potential = xr.DataArray(
        2.0 * sg.EARTH_RADIUS_M * np.cos(latitude) * np.sin(longitude),
        dims=("lat", "lon"),
        coords=coordinates,
        name="vp",
    )
    base_wind = sg.wind(streamfunction, velocity_potential, source="potentials")
    base_u, base_v = base_wind.u, base_wind.v
    scales = np.array([[1.0, -0.5, 2.0], [0.25, 1.5, -1.0]])
    coordinates = {
        "member": ["first", "second"],
        "level": [1000, 850, 500],
        "lat": grid.latitude,
        "lon": grid.longitude,
    }

    def batch(component: xr.DataArray) -> xr.DataArray:
        return xr.DataArray(
            scales[..., None, None] * np.asarray(component),
            dims=("member", "level", "lat", "lon"),
            coords=coordinates,
            name=component.name,
            attrs=component.attrs,
        ).transpose("lat", "member", "lon", "level")

    u, v = batch(base_u), batch(base_v)
    kinematics = sg.kinematics(u, v)
    helmholtz = sg.helmholtz(u, v)
    inverse_gradients = sg.inverse_gradient(u, v)
    wind_from_kinematics = sg.wind(
        kinematics.vo,
        kinematics.d,
        source="vorticity_divergence",
    )
    regridded = sg.regrid_vector(u, v, target, "T3")
    spectral_u, spectral_v = sg.analyze_vector(u, v, "T3").synthesize()

    assert kinematics.vo.dims == u.dims
    assert helmholtz.u_divergent.dims == u.dims
    assert inverse_gradients.dims == u.dims
    assert wind_from_kinematics.u.dims == u.dims
    assert regridded.u.dims == u.dims
    assert spectral_u.dims == u.dims
    assert float(np.max(np.abs(helmholtz.u_divergent))) > 0.0
    assert float(np.max(np.abs(helmholtz.u_rotational))) > 0.0

    for member in range(scales.shape[0]):
        for level in range(scales.shape[1]):
            factor = scales[member, level]
            expected_kinematics = sg.kinematics(base_u * factor, base_v * factor)
            expected_helmholtz = sg.helmholtz(base_u * factor, base_v * factor)
            expected_inverse_gradient = sg.inverse_gradient(
                base_u * factor,
                base_v * factor,
            )
            expected_wind_from_kinematics = sg.wind(
                expected_kinematics.vo,
                expected_kinematics.d,
                source="vorticity_divergence",
            )
            expected_regrid = sg.regrid_vector(
                base_u * factor,
                base_v * factor,
                target,
                "T3",
            )
            expected_spectral_u, expected_spectral_v = sg.analyze_vector(
                base_u * factor,
                base_v * factor,
                "T3",
            ).synthesize()
            xr.testing.assert_allclose(
                kinematics.isel(member=member, level=level, drop=True),
                expected_kinematics,
            )
            xr.testing.assert_allclose(
                helmholtz.isel(member=member, level=level, drop=True),
                expected_helmholtz,
            )
            xr.testing.assert_allclose(
                inverse_gradients.isel(member=member, level=level, drop=True),
                expected_inverse_gradient,
            )
            xr.testing.assert_allclose(
                wind_from_kinematics.u.isel(member=member, level=level, drop=True),
                expected_wind_from_kinematics.u,
            )
            xr.testing.assert_allclose(
                wind_from_kinematics.v.isel(member=member, level=level, drop=True),
                expected_wind_from_kinematics.v,
            )
            xr.testing.assert_allclose(
                regridded.isel(member=member, level=level, drop=True), expected_regrid
            )
            xr.testing.assert_allclose(
                spectral_u.isel(member=member, level=level, drop=True),
                expected_spectral_u,
                rtol=0.0,
                atol=0.0,
            )
            xr.testing.assert_allclose(
                spectral_v.isel(member=member, level=level, drop=True),
                expected_spectral_v,
                rtol=0.0,
                atol=0.0,
            )


def test_detected_horizontal_dimensions_need_not_be_last_or_adjacent() -> None:
    field = scalar_field(supported_grid("cc"), leading=True).transpose(
        "lat", "member", "lon"
    )

    result = sg.filter(field, "T3")

    assert result.dims == field.dims
    np.testing.assert_allclose(result, field, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("batch_shape", [(), (1,), (4,), (2, 3), (2, 3, 4)])
def test_scalar_filter_preserves_arbitrary_leading_shapes(
    batch_shape: tuple[int, ...],
) -> None:
    grid = supported_grid("cc")
    base = scalar_field(grid)
    if batch_shape:
        factors = np.arange(1.0, np.prod(batch_shape) + 1.0).reshape(batch_shape)
        values = factors[..., None, None] * np.asarray(base)
        leading_dims = tuple(f"batch_{index}" for index in range(len(batch_shape)))
        field = xr.DataArray(
            values,
            dims=leading_dims + ("lat", "lon"),
            coords={
                **{
                    dimension: np.arange(size)
                    for dimension, size in zip(leading_dims, batch_shape, strict=True)
                },
                "lat": grid.latitude,
                "lon": grid.longitude,
            },
        )
    else:
        factors = np.asarray(1.0)
        field = base

    result = sg.filter(field, "T2")
    expected_frames = [
        sg.filter(base * float(factor), "T2").values for factor in factors.reshape(-1)
    ]
    expected = np.asarray(expected_frames).reshape(batch_shape + (grid.nlat, grid.nlon))

    assert result.shape == batch_shape + (grid.nlat, grid.nlon)
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=0.0)


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
