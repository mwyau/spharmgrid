"""Dataset xarray accessor discovery and delegation contracts."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from tests.conftest import (
    degree_one_field,
    scalar_field,
    solid_body_wind,
    supported_grid,
)


def _assert_dataset_identical(actual: xr.Dataset, expected: xr.Dataset) -> None:
    """Check Dataset component order, metadata, and values."""
    assert tuple(actual.data_vars) == tuple(expected.data_vars)
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize("kind", ["cc", "gl"])
def test_dataset_grid_accessor_properties(kind: Literal["cc", "gl"]) -> None:
    grid = supported_grid(kind)
    dataset = xr.Dataset({"field": scalar_field(grid)})
    expected = sg.detect_grid(dataset)

    assert dataset.sg.grid.kind == expected.kind
    np.testing.assert_array_equal(dataset.sg.grid.latitude, expected.latitude)
    np.testing.assert_array_equal(dataset.sg.grid.longitude, expected.longitude)
    assert dataset.sg.grid_type == expected.kind


def test_dataset_regrid_vector_accessor() -> None:
    source = supported_grid("cc")
    target = supported_grid("gl")
    u, v = solid_body_wind(source)
    dataset = xr.Dataset({"eastward_input": u, "northward_input": v})

    expected = sg.regrid_vector(
        dataset["eastward_input"],
        dataset["northward_input"],
        target,
        truncation="T2",
        taper=0.1,
        eastward="eastward_output",
        northward="northward_output",
    )
    actual = dataset.sg.regrid_vector(
        target,
        u="eastward_input",
        v="northward_input",
        truncation="T2",
        taper=0.1,
        eastward="eastward_output",
        northward="northward_output",
    )

    _assert_dataset_identical(actual, expected)


def test_dataset_vector_accessors() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    radius = 1.1 * sg.EARTH_RADIUS_M
    dataset = xr.Dataset({"eastward_input": u, "northward_input": v})
    gradient = sg.gradient(
        degree_one_field(grid),
        eastward="east_gradient",
        northward="north_gradient",
        radius=radius,
    )
    dataset = xr.merge((dataset, gradient))
    eastward = dataset["eastward_input"]
    northward = dataset["northward_input"]

    expected = sg.inverse_gradient(
        dataset["east_gradient"],
        dataset["north_gradient"],
        output="potential",
        radius=radius,
    )
    actual = dataset.sg.inverse_gradient(
        eastward="east_gradient",
        northward="north_gradient",
        output="potential",
        radius=radius,
    )
    xr.testing.assert_identical(actual, expected)

    expected = sg.vorticity(
        eastward,
        northward,
        output="relative_vorticity",
        radius=radius,
    )
    actual = dataset.sg.vorticity(
        u="eastward_input",
        v="northward_input",
        output="relative_vorticity",
        radius=radius,
    )
    xr.testing.assert_identical(actual, expected)

    expected = sg.divergence(
        eastward,
        northward,
        output="wind_divergence",
        radius=radius,
    )
    actual = dataset.sg.divergence(
        u="eastward_input",
        v="northward_input",
        output="wind_divergence",
        radius=radius,
    )
    xr.testing.assert_identical(actual, expected)

    expected = sg.kinematics(
        eastward,
        northward,
        vorticity="relative_vorticity",
        divergence="wind_divergence",
        radius=radius,
    )
    actual = dataset.sg.kinematics(
        u="eastward_input",
        v="northward_input",
        vorticity="relative_vorticity",
        divergence="wind_divergence",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.streamfunction(
        eastward,
        northward,
        output="streamfunction",
        radius=radius,
    )
    actual = dataset.sg.streamfunction(
        u="eastward_input",
        v="northward_input",
        output="streamfunction",
        radius=radius,
    )
    xr.testing.assert_identical(actual, expected)

    expected = sg.velocity_potential(
        eastward,
        northward,
        output="velocity_potential",
        radius=radius,
    )
    actual = dataset.sg.velocity_potential(
        u="eastward_input",
        v="northward_input",
        output="velocity_potential",
        radius=radius,
    )
    xr.testing.assert_identical(actual, expected)

    expected = sg.potentials(
        eastward,
        northward,
        streamfunction="streamfunction",
        velocity_potential="velocity_potential",
        radius=radius,
    )
    actual = dataset.sg.potentials(
        u="eastward_input",
        v="northward_input",
        streamfunction="streamfunction",
        velocity_potential="velocity_potential",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.helmholtz(
        eastward,
        northward,
        divergent_eastward="divergent_u",
        divergent_northward="divergent_v",
        rotational_eastward="rotational_u",
        rotational_northward="rotational_v",
        radius=radius,
    )
    actual = dataset.sg.helmholtz(
        u="eastward_input",
        v="northward_input",
        divergent_eastward="divergent_u",
        divergent_northward="divergent_v",
        rotational_eastward="rotational_u",
        rotational_northward="rotational_v",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.vector_laplacian(
        eastward,
        northward,
        eastward="laplacian_u",
        northward="laplacian_v",
        radius=radius,
    )
    actual = dataset.sg.vector_laplacian(
        u="eastward_input",
        v="northward_input",
        eastward="laplacian_u",
        northward="laplacian_v",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)

    laplacian = sg.vector_laplacian(eastward, northward)
    dataset_with_laplacian = xr.Dataset(
        {
            "eastward_input": eastward,
            "northward_input": northward,
            "laplacian_u": laplacian.u,
            "laplacian_v": laplacian.v,
        }
    )
    expected = sg.inverse_vector_laplacian(
        dataset_with_laplacian["laplacian_u"],
        dataset_with_laplacian["laplacian_v"],
        eastward="restored_u",
        northward="restored_v",
        radius=radius,
    )
    actual = dataset_with_laplacian.sg.inverse_vector_laplacian(
        u="laplacian_u",
        v="laplacian_v",
        eastward="restored_u",
        northward="restored_v",
        radius=radius,
    )
    _assert_dataset_identical(actual, expected)


def test_dataset_automatic_component_discovery() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    dataset = xr.Dataset({"ua": u, "va": v})
    dataset.ua.attrs["standard_name"] = "eastward_wind"
    dataset.va.attrs["standard_name"] = "northward_wind"

    expected = sg.kinematics(dataset.ua, dataset.va, vorticity="vort", divergence="div")
    actual = dataset.sg.kinematics(vorticity="vort", divergence="div")

    _assert_dataset_identical(actual, expected)
    assert actual.vort.attrs["standard_name"] == "atmosphere_relative_vorticity"
    assert actual.div.attrs["standard_name"] == "divergence_of_wind"


def test_dataset_scalar_source_selection() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    diagnostics = sg.kinematics(u, v)
    potentials = sg.potentials(u, v)
    sources = xr.Dataset(
        {
            "vo": diagnostics.vo,
            "d": diagnostics.d,
            "strf": potentials.strf,
            "vp": potentials.vp,
        }
    )

    expected = sg.wind(
        diagnostics.vo,
        diagnostics.d,
        eastward="eastward",
        northward="northward",
    )
    actual = xr.Dataset({"vo": diagnostics.vo, "d": diagnostics.d}).sg.wind(
        eastward="eastward",
        northward="northward",
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.wind(
        potentials.strf,
        potentials.vp,
        eastward="eastward",
        northward="northward",
    )
    actual = xr.Dataset({"strf": potentials.strf, "vp": potentials.vp}).sg.wind(
        eastward="eastward",
        northward="northward",
    )
    _assert_dataset_identical(actual, expected)

    actual = sources.sg.wind(
        source="potentials",
        eastward="eastward",
        northward="northward",
    )
    _assert_dataset_identical(actual, expected)

    ambiguous_vorticity = sources.assign(other_vo=diagnostics.vo.rename("other_vo"))
    actual = ambiguous_vorticity.sg.wind(
        source="potentials",
        eastward="eastward",
        northward="northward",
    )
    _assert_dataset_identical(actual, expected)

    expected = sg.rotational_wind(
        diagnostics.vo,
        eastward="rotational_u",
        northward="rotational_v",
    )
    actual = xr.Dataset({"vo": diagnostics.vo}).sg.rotational_wind(
        eastward="rotational_u",
        northward="rotational_v",
    )
    _assert_dataset_identical(actual, expected)

    expected_streamfunction = sg.rotational_wind(
        potentials.strf,
        eastward="rotational_u",
        northward="rotational_v",
    )
    actual = xr.Dataset({"strf": potentials.strf}).sg.rotational_wind(
        eastward="rotational_u",
        northward="rotational_v",
    )
    _assert_dataset_identical(actual, expected_streamfunction)

    scalar_ambiguity = xr.Dataset({"vo": diagnostics.vo, "strf": potentials.strf})
    actual = scalar_ambiguity.sg.rotational_wind(
        field="vo",
        eastward="rotational_u",
        northward="rotational_v",
    )
    _assert_dataset_identical(actual, expected)

    actual = scalar_ambiguity.sg.rotational_wind(
        quantity="vorticity",
        eastward="rotational_u",
        northward="rotational_v",
    )
    _assert_dataset_identical(actual, expected)

    divergent_sources = xr.Dataset({"d": diagnostics.d, "vp": potentials.vp})
    expected = sg.divergent_wind(
        diagnostics.d,
        quantity="divergence",
        eastward="divergent_u",
        northward="divergent_v",
    )
    actual = divergent_sources.sg.divergent_wind(
        field="d",
        eastward="divergent_u",
        northward="divergent_v",
    )
    _assert_dataset_identical(actual, expected)


def test_dataset_source_selection_errors() -> None:
    grid = supported_grid("cc")
    u, v = solid_body_wind(grid)
    diagnostics = sg.kinematics(u, v)
    potentials = sg.potentials(u, v)

    with pytest.raises(ValueError, match="explicit 'u' variable 'missing'"):
        xr.Dataset({"v": v}).sg.vorticity(u="missing")

    with pytest.raises(ValueError, match="could not identify 'v'"):
        xr.Dataset({"u": u}).sg.kinematics()

    duplicate = u.rename("other_u")
    duplicate.attrs["standard_name"] = "eastward_wind"
    u.attrs["standard_name"] = "eastward_wind"
    with pytest.raises(ValueError, match="ambiguous 'u'"):
        xr.Dataset({"u": u, "other_u": duplicate, "v": v}).sg.vorticity()

    with pytest.raises(ValueError, match="requires explicit gradient component"):
        xr.Dataset({"gradient_eastward": u}).sg.inverse_gradient()

    scalar_ambiguity = xr.Dataset({"vo": diagnostics.vo, "strf": potentials.strf})
    with pytest.raises(ValueError, match="both eligible scalar sources"):
        scalar_ambiguity.sg.rotational_wind()

    with pytest.raises(ValueError, match="explicit field 'missing'"):
        scalar_ambiguity.sg.rotational_wind(field="missing")

    with pytest.raises(
        ValueError, match="could not identify an eligible scalar source"
    ):
        xr.Dataset().sg.rotational_wind()

    invalid_quantity = cast(Literal["vorticity", "streamfunction"], "invalid")
    with pytest.raises(ValueError, match="quantity must be one of"):
        xr.Dataset().sg.rotational_wind(quantity=invalid_quantity)

    with pytest.raises(ValueError, match="both vorticity/divergence"):
        xr.Dataset(
            {
                "vo": diagnostics.vo,
                "d": diagnostics.d,
                "strf": potentials.strf,
                "vp": potentials.vp,
            }
        ).sg.wind()

    with pytest.raises(ValueError, match="no complete"):
        xr.Dataset({"vo": diagnostics.vo}).sg.wind()
