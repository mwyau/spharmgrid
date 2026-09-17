"""Optional autograd, dtype, and device checks for the Torch backend."""

from __future__ import annotations

import pytest

import spharmgrid as sg
import spharmgrid.torch as sgt
import spharmgrid.torch.nn as sgnn
from tests.optional.torch.conftest import torch


def test_filter_regrid_and_kinematics_pass_gradcheck() -> None:
    source = sg.gaussian_grid(4, 8, lon0=45.0)
    target = sg.gaussian_grid(5, 10, latitude_order="descending", lon0=-75.0)
    field = torch.linspace(
        -1.0,
        1.0,
        source.nlat * source.nlon,
        dtype=torch.float64,
        requires_grad=True,
    ).reshape(source.nlat, source.nlon)

    assert torch.autograd.gradcheck(
        lambda value: sgt.filter(value, "T2", grid=source),
        (field,),
        eps=1.0e-6,
        atol=1.0e-7,
        rtol=1.0e-5,
    )
    assert torch.autograd.gradcheck(
        lambda value: sgt.regrid(value, target, "T2", source_grid=source),
        (field,),
        eps=1.0e-6,
        atol=1.0e-7,
        rtol=1.0e-5,
    )

    eastward = field.detach().clone().requires_grad_()
    northward = torch.flip(field.detach(), dims=(-2,)).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda u, v: sgt.kinematics(u, v, grid=source)[0],
        (eastward, northward),
        eps=1.0e-6,
        atol=1.0e-7,
        rtol=1.0e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_dtypes_propagate_through_reusable_module(
    dtype: torch.dtype,
) -> None:
    grid = sg.gaussian_grid(6, 12)
    field = torch.ones((grid.nlat, grid.nlon), dtype=dtype, requires_grad=True)
    layer = sgnn.SHTFilter(grid, "T2").to(dtype=dtype)
    result = layer(field)
    assert result.dtype == dtype
    result.square().sum().backward()
    assert field.grad is not None
    assert torch.isfinite(field.grad).all()


def test_unsupported_dtype_is_rejected() -> None:
    grid = sg.gaussian_grid(4, 8)
    with pytest.raises(TypeError, match="float32 or float64"):
        sgt.filter(torch.zeros((grid.nlat, grid.nlon), dtype=torch.float16), grid=grid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_smoke_scalar_vector_autograd_and_reusable_modules(
    dtype: torch.dtype,
) -> None:
    source = sg.gaussian_grid(4, 8, latitude_order="ascending", lon0=45.0)
    target = sg.gaussian_grid(5, 10, latitude_order="descending", lon0=-75.0)
    field = torch.randn(
        source.nlat,
        source.nlon,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    eastward = torch.randn_like(field, requires_grad=True)
    northward = torch.randn_like(field, requires_grad=True)

    functional_filter = sgt.filter(field, "T2", grid=source)
    functional_regrid = sgt.regrid(
        field,
        target,
        "T2",
        source_grid=source,
    )
    functional_u, functional_v = sgt.regrid_vector(
        eastward,
        northward,
        target,
        "T2",
        source_grid=source,
    )
    functional_vorticity, functional_divergence = sgt.kinematics(
        eastward,
        northward,
        grid=source,
    )

    operators = sgnn.SHTOperators(source).to(device="cuda", dtype=dtype)
    filter_layer = sgnn.SHTFilter(source, "T2").to(device="cuda", dtype=dtype)
    regrid_layer = sgnn.SHTRegrid(source, target, "T2").to(
        device="cuda",
        dtype=dtype,
    )
    vector_regrid_layer = sgnn.SHTVectorRegrid(source, target, "T2").to(
        device="cuda",
        dtype=dtype,
    )
    module_filter = filter_layer(field)
    module_vorticity, module_divergence = operators.kinematics(
        eastward,
        northward,
    )
    module_regrid = regrid_layer(field)
    module_u, module_v = vector_regrid_layer(eastward, northward)

    outputs = (
        functional_filter,
        functional_regrid,
        functional_u,
        functional_v,
        functional_vorticity,
        functional_divergence,
        module_filter,
        module_vorticity,
        module_divergence,
        module_regrid,
        module_u,
        module_v,
    )
    assert all(value.is_cuda for value in outputs)
    assert all(value.dtype == dtype for value in outputs)
    assert all(torch.isfinite(value).all() for value in outputs)
    assert all(
        buffer.is_cuda
        for module in (
            operators,
            filter_layer,
            regrid_layer,
            vector_regrid_layer,
        )
        for buffer in module.buffers()
    )

    sum(value.square().mean() for value in outputs).backward()
    for value in (field, eastward, northward):
        assert value.grad is not None
        assert value.grad.is_cuda
        assert value.grad.dtype == dtype
        assert value.grad.shape == value.shape
        assert torch.isfinite(value.grad).all()
