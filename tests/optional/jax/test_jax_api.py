# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public namespace, signatures, and capability checks for JAX."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

import spharmgrid as sg
import spharmgrid.jax as sgj

_FUNCTIONS = {
    "analyze",
    "analyze_vector",
    "filter",
    "regrid",
    "regrid_vector",
    "gradient",
    "inverse_gradient",
    "laplacian",
    "inverse_laplacian",
    "vector_laplacian",
    "inverse_vector_laplacian",
    "vorticity",
    "divergence",
    "kinematics",
    "streamfunction",
    "velocity_potential",
    "potentials",
    "helmholtz",
    "rotational_wind",
    "divergent_wind",
    "wind",
}
_HELPERS = {"device_put", "device_get"}

_XARRAY_ONLY = {
    "analyze": {"sht_threads"},
    "analyze_vector": {"sht_threads"},
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
    for name in _FUNCTIONS
}
_TENSOR_REQUIRED = {
    "rotational_wind": {"source"},
    "divergent_wind": {"source"},
    "wind": {"source"},
}
_SHARED_DEFAULTS = ("radius", "truncation", "lmin", "lmax", "taper")
_REQUIRED = object()


def _signature_parts(
    name: str,
    function: Callable[..., object],
    *,
    tensor: bool,
) -> tuple[tuple[str, inspect._ParameterKind, object], ...]:
    excluded = _TENSOR_ONLY[name] if tensor else _XARRAY_ONLY[name]
    required = _TENSOR_REQUIRED.get(name, set())
    return tuple(
        (
            parameter.name,
            parameter.kind,
            _REQUIRED if parameter.name in required else parameter.default,
        )
        for parameter in inspect.signature(function).parameters.values()
        if parameter.name not in excluded
    )


def test_jax_exports_exactly_the_scientific_operation_set() -> None:
    assert (
        set(sgj.__all__)
        == _FUNCTIONS
        | {
            "SpectralField",
            "SpectralVectorField",
        }
        | _HELPERS
    )
    assert {name for name in sgj.__dict__ if not name.startswith("_")} >= _FUNCTIONS
    assert sgj.SpectralField is not None
    assert sgj.SpectralVectorField is not None
    assert callable(sgj.analyze)
    assert callable(sgj.analyze_vector)
    assert callable(sgj.device_put)
    assert callable(sgj.device_get)
    assert "DataArrayAccessor" not in sgj.__all__
    assert "DatasetAccessor" not in sgj.__all__


def test_accessor_implementation_is_private() -> None:
    assert "DataArrayAccessor" not in sg.__all__
    assert "DatasetAccessor" not in sg.__all__
    assert not hasattr(sg, "DataArrayAccessor")
    assert not hasattr(sg, "DatasetAccessor")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("spharmgrid.accessors")


def test_jax_signatures_match_the_root_scientific_semantics() -> None:
    for name in _FUNCTIONS:
        assert _signature_parts(name, getattr(sg, name), tensor=False) == (
            _signature_parts(name, getattr(sgj, name), tensor=True)
        )
        root_parameters = inspect.signature(getattr(sg, name)).parameters
        jax_parameters = inspect.signature(getattr(sgj, name)).parameters
        for parameter_name in _SHARED_DEFAULTS:
            if parameter_name in root_parameters:
                assert parameter_name in jax_parameters
                assert (
                    root_parameters[parameter_name].default
                    == jax_parameters[parameter_name].default
                )


def test_root_import_does_not_load_optional_jax_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import spharmgrid; assert 'jax' not in sys.modules; "
            "assert 's2fft' not in sys.modules; "
            "assert 'torch' not in sys.modules; "
            "assert 'torch_harmonics' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""


@pytest.mark.parametrize(
    "blocked_name",
    ["jax", "s2fft"],
)
def test_missing_backend_dependency_has_a_targeted_error(blocked_name: str) -> None:
    code = f"""
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == {blocked_name!r} or name.startswith({blocked_name!r} + '.'):
        raise ModuleNotFoundError({blocked_name!r}, name={blocked_name!r})
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
try:
    import spharmgrid.jax
except ImportError as error:
    message = str(error)
    assert 'spharmgrid.jax' in message
    assert {blocked_name!r} in message.lower()
else:
    raise AssertionError('the optional dependency error was not raised')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.jax_x64
def test_regular_gl_accepts_longitude_counts_outside_s2fft_public_shape() -> None:
    shape = (8, 14)
    grid = sg.Grid(
        "gl",
        sg.gaussian_grid(8, 15).latitude,
        np.arange(shape[1], dtype=np.float64) * (360.0 / shape[1]),
    )
    field = jnp.zeros(shape, dtype=jnp.float64)
    assert sgj.filter(field, grid=grid).shape == shape


@pytest.mark.jax_x64
def test_nonstandard_cc_mwss_shape_still_raises_a_capability_error() -> None:
    shape = (9, 15)
    grid = sg.Grid(
        "cc",
        sg.clenshaw_curtis_grid(9, 16).latitude,
        np.arange(shape[1], dtype=np.float64) * (360.0 / shape[1]),
    )
    field = jnp.zeros(shape, dtype=jnp.float64)
    with pytest.raises(ValueError, match="CC/MWSS grids only with shape"):
        sgj.filter(field, grid=grid)
