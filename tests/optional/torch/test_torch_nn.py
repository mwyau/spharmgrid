# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional tests for reusable Torch execution modules."""

from __future__ import annotations

import subprocess
import sys

import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
import spharmgrid.torch.nn as sgnn
from tests.optional.torch.conftest import make_fields, torch


def _assert_tuple_close(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)


def test_torch_nn_exports_match_expected_api() -> None:
    assert sgnn.__all__ == [
        "SHTOperators",
        "SHTFilter",
        "SHTRegrid",
        "SHTVectorRegrid",
    ]


def test_core_import_does_not_load_torch() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import spharmgrid; "
                "assert not any(name in sys.modules for name in "
                "('torch', 'torch_harmonics'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_cc_sht_operators_raise_capability_error(cc_grid: sg.Grid) -> None:
    error = (
        r"torch-harmonics.*CC triangular bands through T8.*"
        r"Full-domain operations.*filter, regrid, and regrid_vector"
    )
    with pytest.raises(ValueError, match=error):
        sgnn.SHTOperators(cc_grid)


def test_sht_operators_match_functional_api(gl_grid: sg.Grid) -> None:
    field, eastward, northward = make_fields(gl_grid)
    operators = sgnn.SHTOperators(gl_grid)

    torch.testing.assert_close(
        operators.filter(field, "T2"),
        sgt.filter(field, "T2", grid=gl_grid),
    )
    expected_gradient = sgt.gradient(field, grid=gl_grid)
    _assert_tuple_close(operators.gradient(field), expected_gradient)

    torch.testing.assert_close(
        operators.inverse_gradient(*expected_gradient),
        sgt.inverse_gradient(*expected_gradient, grid=gl_grid),
    )
    torch.testing.assert_close(
        operators.laplacian(field),
        sgt.laplacian(field, grid=gl_grid),
    )
    torch.testing.assert_close(
        operators.inverse_laplacian(field),
        sgt.inverse_laplacian(field, grid=gl_grid),
    )

    _assert_tuple_close(
        operators.vector_laplacian(eastward, northward),
        sgt.vector_laplacian(eastward, northward, grid=gl_grid),
    )
    _assert_tuple_close(
        operators.inverse_vector_laplacian(eastward, northward),
        sgt.inverse_vector_laplacian(eastward, northward, grid=gl_grid),
    )

    expected_kinematics = sgt.kinematics(eastward, northward, grid=gl_grid)
    _assert_tuple_close(
        operators.kinematics(eastward, northward),
        expected_kinematics,
    )
    torch.testing.assert_close(
        operators.vorticity(eastward, northward),
        expected_kinematics[0],
    )
    torch.testing.assert_close(
        operators.divergence(eastward, northward),
        expected_kinematics[1],
    )

    expected_potentials = sgt.potentials(eastward, northward, grid=gl_grid)
    _assert_tuple_close(
        operators.potentials(eastward, northward),
        expected_potentials,
    )
    torch.testing.assert_close(
        operators.streamfunction(eastward, northward),
        expected_potentials[0],
    )
    torch.testing.assert_close(
        operators.velocity_potential(eastward, northward),
        expected_potentials[1],
    )

    _assert_tuple_close(
        operators.helmholtz(eastward, northward),
        sgt.helmholtz(eastward, northward, grid=gl_grid),
    )

    rotational = operators.rotational_wind(
        expected_kinematics[0],
        source="vorticity",
    )
    expected_rotational = sgt.rotational_wind(
        expected_kinematics[0],
        grid=gl_grid,
        source="vorticity",
    )
    _assert_tuple_close(rotational, expected_rotational)

    divergent = operators.divergent_wind(
        expected_kinematics[1],
        source="divergence",
    )
    expected_divergent = sgt.divergent_wind(
        expected_kinematics[1],
        grid=gl_grid,
        source="divergence",
    )
    _assert_tuple_close(divergent, expected_divergent)

    _assert_tuple_close(
        operators.wind(
            expected_kinematics[0],
            expected_kinematics[1],
            source="vorticity_divergence",
        ),
        sgt.wind(
            expected_kinematics[0],
            expected_kinematics[1],
            grid=gl_grid,
            source="vorticity_divergence",
        ),
    )


def test_sht_operators_reuse_transform_state(gl_grid: sg.Grid) -> None:
    field, _, _ = make_fields(gl_grid)
    operators = sgnn.SHTOperators(gl_grid)
    state = operators._state

    operators.filter(field, "T2")

    assert operators._state is state


def test_sht_operators_preserve_dtype_and_leading_dimensions(
    gl_grid: sg.Grid,
) -> None:
    field, eastward, northward = make_fields(gl_grid)
    operators = sgnn.SHTOperators(gl_grid).to(dtype=torch.float32)

    field32 = field.to(dtype=torch.float32)
    filtered = operators.filter(field32, "T2")
    assert filtered.dtype == torch.float32

    batched_eastward = eastward.to(dtype=torch.float32).unsqueeze(0).expand(2, -1, -1)
    batched_northward = northward.to(dtype=torch.float32).unsqueeze(0).expand(2, -1, -1)
    for result in operators.kinematics(batched_eastward, batched_northward):
        assert result.shape == (2, gl_grid.nlat, gl_grid.nlon)
        assert result.dtype == torch.float32


def test_fixed_transform_modules_match_functional_api(
    gl_grid: sg.Grid,
    cc_grid: sg.Grid,
) -> None:
    field, eastward, northward = make_fields(gl_grid)

    filter_layer = sgnn.SHTFilter(gl_grid, "T2")
    assert isinstance(filter_layer, torch.nn.Module)
    filter_state_id = id(filter_layer._state)
    torch.testing.assert_close(
        filter_layer(field),
        sgt.filter(field, "T2", grid=gl_grid),
    )
    batched_field = field.unsqueeze(0).expand(2, -1, -1)
    torch.testing.assert_close(
        filter_layer(batched_field),
        sgt.filter(batched_field, "T2", grid=gl_grid),
    )
    assert id(filter_layer._state) == filter_state_id

    regrid_layer = sgnn.SHTRegrid(gl_grid, cc_grid, "T2")
    expected_regrid = sgt.regrid(field, cc_grid, "T2", source_grid=gl_grid)
    torch.testing.assert_close(regrid_layer(field), expected_regrid)

    vector_layer = sgnn.SHTVectorRegrid(gl_grid, cc_grid, "T2")
    expected_vector = sgt.regrid_vector(
        eastward,
        northward,
        cc_grid,
        "T2",
        source_grid=gl_grid,
    )
    for actual, expected in zip(
        vector_layer(eastward, northward),
        expected_vector,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_modules_work_inside_a_trainable_torch_model() -> None:
    grid = sg.gaussian_grid(4, 8)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.filter = sgnn.SHTFilter(grid, "T2")
            self.operators = sgnn.SHTOperators(grid)

        def forward(
            self,
            field: torch.Tensor,
            eastward: torch.Tensor,
            northward: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            filtered = self.filter(field)
            vorticity, divergence = self.operators.kinematics(eastward, northward)
            return filtered, vorticity, divergence

    field = (
        torch.linspace(
            -1.0,
            1.0,
            grid.nlat * grid.nlon,
            dtype=torch.float64,
        )
        .reshape(grid.nlat, grid.nlon)
        .requires_grad_()
    )
    eastward = field.detach().clone().requires_grad_()
    northward = torch.flip(field.detach(), dims=(-2,)).requires_grad_()
    output = Model()(field, eastward, northward)
    torch.stack([value.square().mean() for value in output]).sum().backward()
    for value in (field, eastward, northward):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_fixed_filter_module_runs_under_torch_compile_eager() -> None:
    grid = sg.gaussian_grid(4, 8)
    field = (
        torch.linspace(
            -1.0,
            1.0,
            grid.nlat * grid.nlon,
            dtype=torch.float32,
        )
        .reshape(grid.nlat, grid.nlon)
        .requires_grad_()
    )
    layer = sgnn.SHTFilter(grid, "T2")
    compiled = torch.compile(layer, backend="eager")
    torch.testing.assert_close(compiled(field), layer(field))


@pytest.mark.filterwarnings(
    "ignore:Torchinductor does not support code generation for complex "
    "operators.*:UserWarning"
)
@pytest.mark.filterwarnings(
    "ignore:TensorFloat32 tensor cores for float32 matrix multiplication "
    "available but not enabled.*:UserWarning"
)
@pytest.mark.skipif(
    not hasattr(torch, "compile") or not torch.cuda.is_available(),
    reason="real torch.compile CUDA validation requires CUDA",
)
def test_fixed_filter_module_supports_real_torch_compile_on_cuda() -> None:
    grid = sg.gaussian_grid(4, 8)
    field = (
        torch.linspace(
            -1.0,
            1.0,
            grid.nlat * grid.nlon,
            device="cuda",
            dtype=torch.float32,
        )
        .reshape(grid.nlat, grid.nlon)
        .requires_grad_()
    )
    layer = sgnn.SHTFilter(grid, "T2").to(device="cuda", dtype=torch.float32)
    compiled = torch.compile(layer)

    eager = layer(field)
    torch.cuda.synchronize()
    try:
        outputs = []
        for _ in range(2):
            output = compiled(field)
            torch.cuda.synchronize()
            outputs.append(output)
    except RuntimeError as error:

        def _torch_version() -> tuple[int, int]:
            major, minor = torch.__version__.split(".", 2)[:2]
            return int(major), int(minor)

        if not (_torch_version() < (2, 14) and "KeyError: 'complex64'" in str(error)):
            raise
        pytest.xfail(
            "Torch <2.14 CUDA Inductor fails compiling torch-harmonics RealSHT "
            "with KeyError: 'complex64'"
        )

    for output in outputs:
        assert output.is_cuda
        torch.testing.assert_close(output, eager)
    outputs[-1].square().mean().backward()
    assert field.grad is not None
    assert field.grad.is_cuda
    assert field.grad.shape == field.shape
    assert torch.isfinite(field.grad).all()
