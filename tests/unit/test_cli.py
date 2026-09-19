# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

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
    assert "usage: spharmgrid" in capsys.readouterr().out

    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.startswith("spharmgrid ")


def test_cli_execution_options_require_positive_integers() -> None:
    parser = cli._parser()
    arguments = parser.parse_args(
        [
            "filter",
            "input.nc",
            "output.nc",
            "--workers",
            "3",
            "--sht-threads",
            "2",
        ]
    )

    assert arguments.workers == 3
    assert arguments.sht_threads == 2
    with pytest.raises(SystemExit):
        parser.parse_args(["info", "input.nc", "--workers", "2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["filter", "input.nc", "output.nc", "--workers", "0"])


@pytest.mark.parametrize(
    ("available_cpus", "workers", "sht_threads", "expected"),
    [
        (2, None, None, (1, 2)),
        (10, None, None, (3, 4)),
        (10, None, 8, (2, 8)),
        (10, 3, None, (3, 4)),
        (10, 5, 2, (5, 2)),
        (16, None, None, (4, 4)),
        (16, 8, None, (8, 2)),
        (16, None, 2, (8, 2)),
    ],
)
def test_cli_execution_policy_pair_resolution(
    monkeypatch: pytest.MonkeyPatch,
    available_cpus: int,
    workers: int | None,
    sht_threads: int | None,
    expected: tuple[int, int],
) -> None:
    monkeypatch.setattr(cli, "_available_cpu_count", lambda: available_cpus)

    policy = cli._resolve_execution_policy(
        workers=workers,
        sht_threads=sht_threads,
    )

    assert (policy.workers, policy.sht_threads) == expected


def test_cli_available_cpu_count_prefers_process_count_and_affinity_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.os, "process_cpu_count", lambda: 7, raising=False)
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _: {1, 2}, raising=False)
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 3)
    assert cli._available_cpu_count() == 7

    monkeypatch.setattr(cli.os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(cli.os, "sched_getaffinity", lambda _: {1, 2, 3, 4})
    assert cli._available_cpu_count() == 4


def test_import_does_not_load_optional_cli_backends() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import spharmgrid; "
                "assert not any(name in sys.modules for name in "
                "('dask', 'h5netcdf', 'zarr', 'cfgrib'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


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
