# SPDX-FileCopyrightText: Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dask-backed execution, Xarray, and CLI integration tests."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid import cli, spectral
from spharmgrid.cli import main
from tests.conftest import scalar_field, solid_body_wind, supported_grid

dask = pytest.importorskip("dask")


def _record_scalar_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, Mock]:
    analysis = Mock(wraps=spectral.scalar_analysis)
    synthesis = Mock(wraps=spectral.scalar_synthesis)
    monkeypatch.setattr(spectral, "scalar_analysis", analysis)
    monkeypatch.setattr(spectral, "scalar_synthesis", synthesis)
    return analysis, synthesis


def _threads(*mocks: Mock) -> list[int]:
    return [call.kwargs["nthreads"] for mock in mocks for call in mock.call_args_list]


def test_dask_default_uses_one_thread_and_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = scalar_field(supported_grid("cc"), leading=True).chunk(
        {"member": 1, "lat": 8, "lon": 12}
    )
    mocks = _record_scalar_threads(monkeypatch)

    with dask.config.set(pool=None, num_workers=None):
        result = sg.filter(field, "T2")
        assert hasattr(result.data, "dask")
        result.compute()

    assert _threads(*mocks) == [1, 1, 1, 1]


def test_dask_api_leaves_caller_scheduler_settings_unchanged() -> None:
    field = scalar_field(supported_grid("cc"), leading=True).chunk(
        {"member": 1, "lat": 8, "lon": 12}
    )
    caller_pool = object()

    with dask.config.set(pool=caller_pool, num_workers=7):
        result = sg.filter(field, "T2", sht_threads=8)

        assert hasattr(result.data, "dask")
        assert dask.config.get("pool") is caller_pool
        assert dask.config.get("num_workers") == 7


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
    xr.testing.assert_allclose(laplacian.compute(), eager_laplacian, rtol=0.0, atol=0.0)
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


def test_cli_uses_its_executor_through_lazy_output_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = scalar_field(supported_grid("cc"), name="msl", leading=True).chunk(
        {"member": 1, "lat": 8, "lon": 12}
    )
    opened = field.to_dataset()
    open_chunks: list[str | None] = []
    observed_pools: list[object] = []
    observed_sht_threads: list[int] = []

    def fake_open_dataset(
        path: str, *, chunks: str | None = None
    ) -> AbstractContextManager[xr.Dataset]:
        open_chunks.append(chunks)
        return cast(AbstractContextManager[xr.Dataset], nullcontext(opened))

    def fake_write_dataset(dataset: xr.Dataset, output: str) -> None:
        pool = dask.config.get("pool")
        assert pool is not None
        observed_pools.append(pool)
        assert all(
            hasattr(variable.data, "dask") for variable in dataset.data_vars.values()
        )
        dataset.compute()
        assert dask.config.get("pool") is pool

    def fake_filter(*args: object, **kwargs: object) -> xr.DataArray:
        observed_sht_threads.append(cast(int, kwargs["sht_threads"]))
        return field.isel(member=0)

    monkeypatch.setattr(cli, "_open_dataset", fake_open_dataset)
    monkeypatch.setattr(cli, "_write_dataset", fake_write_dataset)
    monkeypatch.setattr(cli, "filter", Mock(side_effect=fake_filter))
    missing = object()

    with dask.config.set(pool=None, num_workers=None):
        assert (
            main(
                [
                    "filter",
                    "input.nc",
                    "output.nc",
                    "--var",
                    "msl",
                    "--workers",
                    "2",
                    "--sht-threads",
                    "1",
                ]
            )
            == 0
        )
        assert dask.config.get("pool", default=missing) is None
        assert dask.config.get("num_workers", default=missing) is None

    assert open_chunks == ["auto"]
    assert len(observed_pools) == 1
    assert observed_sht_threads == [1]
