"""Backend-independent command-line interface tests."""

from __future__ import annotations

import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
import xarray as xr

from spharmgrid import cli
from spharmgrid.cli import main
from tests.conftest import scalar_field, supported_grid


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


def test_cli_uses_its_executor_through_lazy_output_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dask = pytest.importorskip("dask")
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
