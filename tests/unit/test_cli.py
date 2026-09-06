"""Backend-independent command-line interface tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import xarray as xr

from spharmgrid import cli
from spharmgrid.cli import main


def test_core_cli_help_and_version_do_not_need_file_backends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "NetCDF and Zarr read/write plus GRIB input" in capsys.readouterr().out

    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.startswith("spharmgrid ")


def test_import_does_not_load_optional_cli_backends() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import spharmgrid; "
                "assert not any(name in sys.modules for name in "
                "('h5netcdf', 'zarr', 'cfgrib'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""


def test_missing_cli_dependencies_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fail_open_dataset(path: str) -> xr.Dataset:
        raise ImportError(f"backend unavailable for {path}")

    monkeypatch.setattr(cli.xr, "open_dataset", fail_open_dataset)

    assert main(["info", str(tmp_path / "input.nc")]) == 2
    error = capsys.readouterr().err
    assert "CLI dependencies are not installed." in error
    assert 'pip install "spharmgrid[cli]"' in error
    assert "Traceback" not in error
