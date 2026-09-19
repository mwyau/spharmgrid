# SPDX-FileCopyrightText: Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Threading tests for per-operation spherical harmonic transforms."""

from __future__ import annotations

from importlib import import_module
from typing import cast
from unittest.mock import Mock

import pytest
import xarray as xr

import spharmgrid as sg
from spharmgrid import spectral
from tests.conftest import scalar_field, supported_grid


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


def test_eager_default_uses_ducc_default_thread_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = scalar_field(supported_grid("cc"))
    mocks = _record_scalar_threads(monkeypatch)

    sg.filter(field, "T2")

    assert _threads(*mocks) == [0, 0]


@pytest.mark.parametrize("sht_threads", [1, 2, 8])
def test_explicit_threads_reach_each_scalar_transform(
    monkeypatch: pytest.MonkeyPatch, sht_threads: int
) -> None:
    field = scalar_field(supported_grid("gl"))
    mocks = _record_scalar_threads(monkeypatch)

    sg.filter(field, "T2", sht_threads=sht_threads)

    assert _threads(*mocks) == [sht_threads, sht_threads]


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5, "1"])
def test_invalid_sht_threads_are_rejected(invalid: object) -> None:
    field = scalar_field(supported_grid("cc"))

    with pytest.raises((TypeError, ValueError), match="sht_threads"):
        sg.filter(field, "T2", sht_threads=cast(int | None, invalid))


def test_regrid_propagates_explicit_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regrid_module = import_module("spharmgrid.regrid")
    analysis = Mock(wraps=regrid_module.scalar_analysis)
    synthesis = Mock(wraps=regrid_module.scalar_synthesis)
    monkeypatch.setattr(regrid_module, "scalar_analysis", analysis)
    monkeypatch.setattr(regrid_module, "scalar_synthesis", synthesis)
    field = scalar_field(supported_grid("cc"))

    sg.regrid(field, supported_grid("gl"), "T2", sht_threads=8)

    assert _threads(analysis, synthesis) == [8, 8]


def test_accessor_and_direct_filter_share_thread_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = scalar_field(supported_grid("cc"))
    mocks = _record_scalar_threads(monkeypatch)

    direct = sg.filter(field, "T2", sht_threads=2)
    direct_calls = _threads(*mocks)
    for mock in mocks:
        mock.reset_mock()
    accessor = field.sg.filter("T2", sht_threads=2)

    xr.testing.assert_identical(accessor, direct)
    assert direct_calls == [2, 2]
    assert _threads(*mocks) == [2, 2]
