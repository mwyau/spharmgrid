"""Optional xarray file-backend checks for the CLI dispatch boundary."""

from __future__ import annotations

import sys

import pytest
import xarray as xr

from spharmgrid.cli import _open_dataset


@pytest.mark.skipif(sys.platform != "linux", reason="cfgrib lane is Linux-only")
def test_grib_input_uses_xarray_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI leaves non-Zarr input selection to xarray and cfgrib."""
    pytest.importorskip("cfgrib")
    assert "cfgrib" in xr.backends.list_engines()

    opened = xr.Dataset()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def fake_open_dataset(path: str, *args: object, **kwargs: object) -> xr.Dataset:
        calls.append((path, args, kwargs))
        return opened

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    assert _open_dataset("input.grib") is opened
    assert calls == [("input.grib", (), {})]
