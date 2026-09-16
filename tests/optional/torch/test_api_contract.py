"""Keep the tensor functional namespace aligned with the root API."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import ModuleType

import pytest

import spharmgrid as sg

torch = pytest.importorskip("torch")
sgt = pytest.importorskip("spharmgrid.torch")  # noqa: E402

from spharmgrid._api_contract import FUNCTIONS  # noqa: E402

# These parameters describe information available from Xarray metadata or
# output containers but absent from tensor calls.  They are the only
# namespace-specific signature differences removed before comparison.
_XARRAY_ONLY = {
    "filter": {"sht_threads"},
    "regrid": {"sht_threads"},
    "regrid_vector": {"eastward", "northward", "sht_threads"},
    "gradient": {"eastward", "northward", "sht_threads"},
    "inverse_gradient": {"output", "sht_threads"},
    "laplacian": {"sht_threads"},
    "inverse_laplacian": {"sht_threads"},
    "vector_laplacian": {"eastward", "northward", "sht_threads"},
    "inverse_vector_laplacian": {"eastward", "northward", "sht_threads"},
    "vorticity": {"output", "sht_threads"},
    "divergence": {"output", "sht_threads"},
    "kinematics": {"vorticity", "divergence", "sht_threads"},
    "streamfunction": {"output", "sht_threads"},
    "velocity_potential": {"output", "sht_threads"},
    "potentials": {"streamfunction", "velocity_potential", "sht_threads"},
    "helmholtz": {
        "divergent_eastward",
        "divergent_northward",
        "rotational_eastward",
        "rotational_northward",
        "sht_threads",
    },
    "rotational_wind": {"eastward", "northward", "sht_threads"},
    "divergent_wind": {"eastward", "northward", "sht_threads"},
    "wind": {"eastward", "northward", "sht_threads"},
}

_TENSOR_ONLY = {
    name: {"source_grid" if name in {"regrid", "regrid_vector"} else "grid"}
    for name in FUNCTIONS
}

# Tensor inputs have no CF metadata from which to infer these choices.
_TENSOR_REQUIRED = {
    "rotational_wind": {"quantity"},
    "divergent_wind": {"quantity"},
    "wind": {"source"},
}

_SHARED_DEFAULTS = ("radius", "truncation", "lmin", "lmax", "taper")
_REQUIRED = object()
_ROOT_NON_EXECUTION_FUNCTIONS = {
    "clenshaw_curtis_grid",
    "detect_grid",
    "gaussian_grid",
    "parse_spectral",
}


def _signature_parts(
    function_name: str,
    function: Callable[..., object],
    *,
    tensor: bool,
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    excluded = (_TENSOR_ONLY if tensor else _XARRAY_ONLY)[function_name]
    required = _TENSOR_REQUIRED.get(function_name, set())
    return tuple(
        (
            parameter.name,
            parameter.kind,
            _REQUIRED if parameter.name in required else parameter.default,
        )
        for parameter in inspect.signature(function).parameters.values()
        if parameter.name not in excluded
    )


def _assert_api_parity(
    xarray_namespace: ModuleType,
    tensor_namespace: ModuleType,
) -> None:
    missing_xarray = [name for name in FUNCTIONS if not hasattr(xarray_namespace, name)]
    missing_tensor = [name for name in FUNCTIONS if not hasattr(tensor_namespace, name)]
    assert not missing_xarray, f"missing root functions: {missing_xarray}"
    assert not missing_tensor, f"missing tensor functions: {missing_tensor}"

    root_functions = {
        name
        for name in xarray_namespace.__all__
        if inspect.isfunction(getattr(xarray_namespace, name))
    }
    tensor_functions = {
        name
        for name in tensor_namespace.__all__
        if inspect.isfunction(getattr(tensor_namespace, name))
    }
    assert root_functions == set(FUNCTIONS) | _ROOT_NON_EXECUTION_FUNCTIONS
    assert tensor_functions == set(FUNCTIONS)

    for name in FUNCTIONS:
        assert name in xarray_namespace.__all__
        assert name in tensor_namespace.__all__
        assert _signature_parts(
            name,
            getattr(xarray_namespace, name),
            tensor=False,
        ) == _signature_parts(name, getattr(tensor_namespace, name), tensor=True)

        xarray_parameters = inspect.signature(
            getattr(xarray_namespace, name)
        ).parameters
        tensor_parameters = inspect.signature(
            getattr(tensor_namespace, name)
        ).parameters
        for parameter_name in _SHARED_DEFAULTS:
            if parameter_name not in xarray_parameters:
                continue
            assert parameter_name in tensor_parameters
            assert (
                xarray_parameters[parameter_name].default
                == tensor_parameters[parameter_name].default
            )


def test_torch_functional_api_matches_root_contract() -> None:
    _assert_api_parity(sg, sgt)


def test_parity_guard_detects_missing_function(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sgt, "wind")
    with pytest.raises(AssertionError, match="missing tensor functions"):
        _assert_api_parity(sg, sgt)


def test_parity_guard_detects_missing_shared_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sgt.gradient

    def without_radius(
        field: torch.Tensor,
        *,
        grid: sg.Grid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return original(field, grid=grid)

    monkeypatch.setattr(sgt, "gradient", without_radius)
    with pytest.raises(AssertionError):
        _assert_api_parity(sg, sgt)
