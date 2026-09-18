"""Keep the tensor functional namespace aligned with the root API."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import ModuleType

import spharmgrid as sg
import spharmgrid.torch as sgt

# These parameters describe information available from Xarray metadata or
# output containers but absent from tensor calls. They are the only
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

_ROOT_NON_EXECUTION_NAMES = {
    "clenshaw_curtis_grid",
    "detect_grid",
    "gaussian_grid",
    "parse_spectral",
}

_ROOT_EXECUTION_NAMES = {
    name
    for name in sg.__all__
    if name not in _ROOT_NON_EXECUTION_NAMES and inspect.isfunction(getattr(sg, name))
}

_TENSOR_ONLY = {
    name: {"source_grid" if name in {"regrid", "regrid_vector"} else "grid"}
    for name in _ROOT_EXECUTION_NAMES
}

# Tensor inputs have no CF metadata from which to infer these choices.
_TENSOR_REQUIRED = {
    "rotational_wind": {"source"},
    "divergent_wind": {"source"},
    "wind": {"source"},
}

_SHARED_DEFAULTS = ("radius", "truncation", "lmin", "lmax", "taper")
_REQUIRED = object()


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


def _execution_functions(namespace: ModuleType) -> set[str]:
    return {
        name
        for name in namespace.__all__
        if inspect.isfunction(getattr(namespace, name, None))
    }


def _assert_api_parity(
    xarray_namespace: ModuleType,
    tensor_namespace: ModuleType,
) -> None:
    root_functions = _execution_functions(xarray_namespace)
    tensor_functions = _execution_functions(tensor_namespace)
    execution_functions = root_functions - _ROOT_NON_EXECUTION_NAMES

    assert root_functions == _ROOT_EXECUTION_NAMES | _ROOT_NON_EXECUTION_NAMES
    assert tensor_functions == execution_functions

    for name in execution_functions:
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


def test_torch_functional_api_matches_root_api() -> None:
    _assert_api_parity(sg, sgt)
