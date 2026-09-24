# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch execution helpers backed by ``torch-harmonics``.

The adapter translates spharmgrid's rectangular grid and coefficient
conventions to the native tensors used by torch-harmonics.  It does not expose
the backend's coefficient representation as part of the public API.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch_harmonics as _torch_harmonics

from .._transform import TransformSpec
from ..grids import Grid, grid_capabilities, grid_layout
from ..spectral import (
    _spectral_selection_is_within,
    _validate_taper,
    _validate_vector_spec,
    resolve_transform_spec,
)
from ._cc_sht import _ExtendedCCRealSHT, _ExtendedCCRealVectorSHT


def _validate_grid(grid: Grid, name: str = "grid") -> None:
    if not isinstance(grid, Grid):
        raise TypeError(f"{name} must be a spharmgrid.Grid")


def _resolve_transform_spec(
    source: Grid,
    target: Grid,
    selection: TransformSpec | None,
) -> TransformSpec:
    """Resolve a Torch transform domain and require triangular coefficients."""
    _validate_grid(source, "source_grid")
    _validate_grid(target, "target_grid")
    _validate_torch_selection(selection)
    requested = resolve_transform_spec(source, target, selection)

    if requested.lmax != requested.mmax:
        raise ValueError(
            "spharmgrid.torch supports only triangular coefficient domains; "
            "truncation=None requests a non-triangular "
            "full bandwidth on these grids, so supply an explicit supported Tn"
        )
    return requested


def _intersect_transform_spec(
    target: Grid,
    analyzed: TransformSpec,
) -> TransformSpec:
    """Restrict an analyzed triangular domain to the target grid."""
    limit = grid_capabilities(target).triangular_lmax
    lmax = min(analyzed.lmax, limit)
    return TransformSpec(min(analyzed.lmin, lmax), lmax, lmax)


def _torch_grid_name(grid: Grid) -> str:
    return "legendre-gauss" if grid.kind == "gl" else "equiangular"


def _make_analysis_module(
    source: Grid,
    spec: TransformSpec,
    *,
    vector: bool,
) -> torch.nn.Module:
    """Build the forward analysis module for a grid and spectral domain."""
    lmax = spec.lmax + 1
    mmax = spec.mmax + 1
    if source.kind == "cc" and spec.lmax > (source.nlat - 1) // 2:
        module_type = _ExtendedCCRealVectorSHT if vector else _ExtendedCCRealSHT
        return module_type(source.nlat, source.nlon, lmax=lmax, mmax=mmax)
    module_type = _torch_harmonics.RealVectorSHT if vector else _torch_harmonics.RealSHT
    return module_type(
        source.nlat,
        source.nlon,
        lmax=lmax,
        mmax=mmax,
        grid=_torch_grid_name(source),
        norm="ortho",
        csphase=True,
    )


def _make_synthesis_module(
    target: Grid,
    spec: TransformSpec,
    *,
    vector: bool,
) -> torch.nn.Module:
    """Build the native torch-harmonics inverse transform."""
    lmax = spec.lmax + 1
    mmax = spec.mmax + 1
    module_type = (
        _torch_harmonics.InverseRealVectorSHT
        if vector
        else _torch_harmonics.InverseRealSHT
    )
    return module_type(
        target.nlat,
        target.nlon,
        lmax=lmax,
        mmax=mmax,
        grid=_torch_grid_name(target),
        norm="ortho",
        csphase=True,
    )


def _require_tensor(
    field: torch.Tensor,
    grid: Grid,
    name: str = "field",
) -> None:
    if not isinstance(field, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if field.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if tuple(field.shape[-2:]) != (grid.nlat, grid.nlon):
        raise ValueError(
            f"{name} must end with ({grid.nlat}, {grid.nlon}) spatial dimensions; "
            f"got {tuple(field.shape[-2:])}"
        )
    if field.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"{name} must use float32 or float64; torch-harmonics dtypes "
            f"{field.dtype} are not part of spharmgrid.torch's supported API"
        )


def _require_vector_tensors(
    u: torch.Tensor,
    v: torch.Tensor,
    grid: Grid,
) -> None:
    _require_tensor(u, grid, "u")
    _require_tensor(v, grid, "v")
    if u.shape != v.shape:
        raise ValueError(
            f"u and v must have the same shape; got {u.shape} and {v.shape}"
        )
    if u.dtype != v.dtype:
        raise TypeError(
            f"u and v must have the same dtype; got {u.dtype} and {v.dtype}"
        )
    if u.device != v.device:
        raise ValueError(
            f"u and v must be on the same device; got {u.device} and {v.device}"
        )


def _validate_radius(radius: float) -> None:
    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError("radius must be a positive finite number in metres")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number in metres")


class _TorchTransform(torch.nn.Module):
    """Reusable scalar/vector transform state for one source/target pair."""

    def __init__(
        self,
        source: Grid,
        target: Grid,
        spec: TransformSpec,
        *,
        scalar_analysis: bool = False,
        scalar_synthesis: bool = False,
        vector_analysis: bool = False,
        vector_synthesis: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if spec.lmax != spec.mmax:
            raise ValueError("torch-harmonics transform state must be triangular")
        if not any(
            (scalar_analysis, scalar_synthesis, vector_analysis, vector_synthesis)
        ):
            raise ValueError("transform state must request at least one direction")

        self.source_grid: Grid = source
        self.target_grid: Grid = target
        self.spec: TransformSpec = spec

        source_layout = grid_layout(source)
        target_layout = grid_layout(target)
        modes = torch.arange(spec.mmax + 1, dtype=dtype)
        source_angle = modes * source_layout.phi0_radians
        target_angle = modes * target_layout.phi0_radians
        needs_analysis = scalar_analysis or vector_analysis
        needs_synthesis = scalar_synthesis or vector_synthesis
        self.register_buffer(
            "_source_to_zero_phase_real",
            torch.cos(source_angle) if needs_analysis else None,
            persistent=False,
        )
        self.register_buffer(
            "_source_to_zero_phase_imag",
            -torch.sin(source_angle) if needs_analysis else None,
            persistent=False,
        )
        self.register_buffer(
            "_target_from_zero_phase_real",
            torch.cos(target_angle) if needs_synthesis else None,
            persistent=False,
        )
        self.register_buffer(
            "_target_from_zero_phase_imag",
            torch.sin(target_angle) if needs_synthesis else None,
            persistent=False,
        )
        self.register_buffer(
            "_source_latitude_indices",
            torch.as_tensor(source_layout.latitude.canonical_indices, dtype=torch.long)
            if needs_analysis
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_source_longitude_indices",
            torch.as_tensor(source_layout.longitude.canonical_indices, dtype=torch.long)
            if needs_analysis
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_target_latitude_restore",
            torch.as_tensor(target_layout.latitude.restore_indices, dtype=torch.long)
            if needs_synthesis
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_target_longitude_restore",
            torch.as_tensor(target_layout.longitude.restore_indices, dtype=torch.long)
            if needs_synthesis
            else None,
            persistent=False,
        )
        self.register_buffer(
            "_degrees",
            torch.arange(spec.lmax + 1, dtype=dtype).reshape(-1, 1),
            persistent=False,
        )

        self._scalar_analysis = (
            _make_analysis_module(source, spec, vector=False)
            if scalar_analysis
            else None
        )
        self._scalar_synthesis = (
            _make_synthesis_module(target, spec, vector=False)
            if scalar_synthesis
            else None
        )
        self._vector_analysis = (
            _make_analysis_module(source, spec, vector=True)
            if vector_analysis
            else None
        )
        self._vector_synthesis = (
            _make_synthesis_module(target, spec, vector=True)
            if vector_synthesis
            else None
        )
        self.to(dtype=dtype)

    def _move_to(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _TorchTransform:
        self.to(device=device, dtype=dtype)
        return self

    @property
    def _degree_scale(self) -> torch.Tensor:
        degrees = self._degrees_tensor
        return torch.sqrt(degrees * (degrees + 1.0))

    @property
    def _degrees_tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self._degrees)

    def _check_device(self, field: torch.Tensor) -> None:
        latitude_indices = cast(torch.Tensor, self._source_latitude_indices)
        if latitude_indices.device != field.device:
            raise ValueError(
                "transform state and input tensor must be on the same device; "
                "move the SHT module with .to(input.device)"
            )

    def _check_dtype(self, values: torch.Tensor) -> None:
        if values.dtype != self._degrees_tensor.dtype:
            raise TypeError(
                "transform state and input tensor must use the same dtype; "
                f"move the SHT module with .to(dtype={values.dtype})"
            )

    def _canonicalize(self, field: torch.Tensor) -> torch.Tensor:
        self._check_device(field)
        latitude_indices = cast(torch.Tensor, self._source_latitude_indices)
        longitude_indices = cast(torch.Tensor, self._source_longitude_indices)
        result = field.index_select(-2, latitude_indices)
        return result.index_select(-1, longitude_indices)

    def _restore_target(self, field: torch.Tensor) -> torch.Tensor:
        latitude_indices = cast(torch.Tensor, self._target_latitude_restore)
        longitude_indices = cast(torch.Tensor, self._target_longitude_restore)
        result = field.index_select(-2, latitude_indices)
        return result.index_select(-1, longitude_indices)

    def _phase(
        self,
        real_part: torch.Tensor,
        imaginary_part: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        real_dtype = coefficients.real.dtype
        return torch.complex(
            real_part.to(device=coefficients.device, dtype=real_dtype),
            imaginary_part.to(device=coefficients.device, dtype=real_dtype),
        )

    def scalar_analysis(self, field: torch.Tensor) -> torch.Tensor:
        """Analyze a field and return zero-origin scalar coefficients."""
        if self._scalar_analysis is None:
            raise RuntimeError("scalar analysis direction was not requested")
        self._check_dtype(field)
        canonical = self._canonicalize(field)
        coefficients = cast(torch.Tensor, self._scalar_analysis(canonical))
        phase = self._phase(
            cast(torch.Tensor, self._source_to_zero_phase_real),
            cast(torch.Tensor, self._source_to_zero_phase_imag),
            coefficients,
        )
        return coefficients * phase.reshape(1, -1)

    def scalar_synthesis(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Synthesize zero-origin scalar coefficients in target order."""
        if self._scalar_synthesis is None:
            raise RuntimeError("scalar synthesis direction was not requested")
        self._check_dtype(coefficients.real)
        phase = self._phase(
            cast(torch.Tensor, self._target_from_zero_phase_real),
            cast(torch.Tensor, self._target_from_zero_phase_imag),
            coefficients,
        )
        values = self._scalar_synthesis(coefficients * phase.reshape(1, -1))
        return self._restore_target(values)

    def vector_analysis(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Analyze geographic wind and return DUCC-like E/B coefficients."""
        if self._vector_analysis is None:
            raise RuntimeError("vector analysis direction was not requested")
        self._check_dtype(u)
        self._check_dtype(v)
        canonical_u = self._canonicalize(u)
        canonical_v = self._canonicalize(v)
        vector_map = torch.stack((-canonical_v, canonical_u), dim=-3)
        coefficients = cast(torch.Tensor, self._vector_analysis(vector_map))
        phase = self._phase(
            cast(torch.Tensor, self._source_to_zero_phase_real),
            cast(torch.Tensor, self._source_to_zero_phase_imag),
            coefficients,
        )
        coefficients = coefficients * phase.reshape(1, 1, -1)
        scale = self._degree_scale.to(
            device=coefficients.device, dtype=coefficients.real.dtype
        )
        # The DUCC spin-1 coefficients use E/B normalization.  The verified
        # torch-harmonics conversion is E=sqrt(l(l+1))*s and
        # B=-sqrt(l(l+1))*t for positive degree.
        return torch.stack(
            (
                coefficients.select(-3, 0) * scale,
                -coefficients.select(-3, 1) * scale,
            ),
            dim=-3,
        )

    def vector_synthesis(
        self, coefficients: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesize DUCC-like E/B coefficients as geographic wind."""
        if self._vector_synthesis is None:
            raise RuntimeError("vector synthesis direction was not requested")
        self._check_dtype(coefficients.real)
        scale = self._degree_scale.to(
            device=coefficients.device, dtype=coefficients.real.dtype
        )
        positive = scale > 0.0
        safe_scale = torch.where(positive, scale, torch.ones_like(scale))
        spheroidal = torch.where(
            positive,
            coefficients.select(-3, 0) / safe_scale,
            torch.zeros_like(coefficients.select(-3, 0)),
        )
        toroidal = torch.where(
            positive,
            -coefficients.select(-3, 1) / safe_scale,
            torch.zeros_like(coefficients.select(-3, 1)),
        )
        torch_coefficients = torch.stack((spheroidal, toroidal), dim=-3)
        phase = self._phase(
            cast(torch.Tensor, self._target_from_zero_phase_real),
            cast(torch.Tensor, self._target_from_zero_phase_imag),
            torch_coefficients,
        )
        values = self._vector_synthesis(torch_coefficients * phase.reshape(1, 1, -1))
        return (
            self._restore_target(values.select(-3, 1)),
            self._restore_target(-values.select(-3, 0)),
        )

    def _discard_analysis(self) -> None:
        """Release source-side state after reusable coefficients are available."""
        self._scalar_analysis = None
        self._vector_analysis = None
        self._source_latitude_indices = None
        self._source_longitude_indices = None
        self._source_to_zero_phase_real = None
        self._source_to_zero_phase_imag = None


def _spectral_weights(
    state: _TorchTransform,
    selection: TransformSpec,
    taper: float | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a Torch-native degree response for a spectral selection."""
    if selection.lmax > state.spec.lmax:
        raise ValueError(
            f"requested lmax={selection.lmax} exceeds transform lmax={state.spec.lmax}"
        )
    _validate_taper(taper)
    degrees = cast(torch.Tensor, state._degrees).to(dtype=dtype)
    inside = (degrees >= selection.lmin) & (degrees <= selection.lmax)
    if taper is None:
        response = torch.ones_like(degrees)
    elif selection.lmax == 0:
        response = torch.full_like(degrees, float(taper))
    else:
        coefficient = (
            -math.log(float(taper)) / (selection.lmax * (selection.lmax + 1)) ** 2
        )
        response = torch.exp(-coefficient * (degrees * (degrees + 1.0)) ** 2)
    return torch.where(inside, response, torch.zeros_like(response))


def _apply_selection(
    coefficients: torch.Tensor,
    state: _TorchTransform,
    selection: TransformSpec,
    taper: float | None,
) -> torch.Tensor:
    weights = _spectral_weights(state, selection, taper, coefficients.real.dtype)
    return coefficients * weights


def _make_state(
    source: Grid,
    target: Grid,
    selection: TransformSpec | None,
    *,
    scalar_analysis: bool = False,
    scalar_synthesis: bool = False,
    vector_analysis: bool = False,
    vector_synthesis: bool = False,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> _TorchTransform:
    spec = _resolve_transform_spec(source, target, selection)
    state = _TorchTransform(
        source,
        target,
        spec,
        scalar_analysis=scalar_analysis,
        scalar_synthesis=scalar_synthesis,
        vector_analysis=vector_analysis,
        vector_synthesis=vector_synthesis,
        dtype=dtype,
    )
    return state._move_to(device, dtype)


def _check_selection(
    selection: TransformSpec | None,
    state: _TorchTransform,
) -> None:
    _validate_torch_selection(selection)
    if selection is None:
        return
    if not _spectral_selection_is_within(state.spec, selection):
        raise ValueError(
            f"requested spectral selection {selection} exceeds the current "
            f"spectral domain {state.spec}; discarded modes cannot be restored"
        )


def _validate_torch_selection(selection: TransformSpec | None) -> None:
    """Reject parsed spectral shapes not yet implemented by torch-harmonics."""
    if selection is None or selection.truncation == "triangular":
        return
    if selection.truncation == "trapezoidal":
        notation = f"T{selection.lmax}x{selection.mmax}"
    else:
        notation = f"R{selection.lmax - selection.mmax}"
    raise NotImplementedError(
        "torch-harmonics backend support for "
        f"{selection.truncation} truncation {notation} is not enabled yet"
    )


def _require_vector_bandwidth(state: _TorchTransform) -> None:
    _validate_vector_spec(state.spec)
