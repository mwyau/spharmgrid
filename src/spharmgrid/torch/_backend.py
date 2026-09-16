"""PyTorch execution helpers backed by ``torch-harmonics``.

The adapter translates spharmgrid's rectangular grid and coefficient
conventions to the native tensors used by torch-harmonics.  It does not expose
the backend's coefficient representation as part of the public API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch_harmonics as _torch_harmonics

from .._transform import TransformSpec
from ..grids import Grid, grid_layout
from ..spectral import SpectralRange, _validate_taper, transform_spec


@dataclass(frozen=True, slots=True)
class _TorchGridCapabilities:
    """Conservative bandwidth limits demonstrated for torch-harmonics."""

    latitude_lmax: int
    longitude_mmax: int

    @property
    def triangular_lmax(self) -> int:
        """Largest inclusive total degree in a triangular transform."""
        return min(self.latitude_lmax, self.longitude_mmax)


def _validate_grid(grid: Grid, name: str = "grid") -> None:
    if not isinstance(grid, Grid):
        raise TypeError(f"{name} must be a spharmgrid.Grid")


def _torch_capabilities(grid: Grid) -> _TorchGridCapabilities:
    """Return the verified spharmgrid-compatible torch-harmonics limits."""
    latitude_lmax = grid.nlat - 1 if grid.kind == "gl" else (grid.nlat - 1) // 2
    # spharmgrid intentionally excludes the real-FFT Nyquist mode when nlon is
    # even, matching DUCC's ``(nphi - 1) // 2`` limit.
    longitude_mmax = (grid.nlon - 1) // 2
    return _TorchGridCapabilities(latitude_lmax, longitude_mmax)


def _resolve_transform_spec(
    source: Grid,
    target: Grid,
    selection: SpectralRange | None,
) -> TransformSpec:
    """Resolve a transform without changing the requested coefficient domain."""
    _validate_grid(source, "source_grid")
    _validate_grid(target, "target_grid")
    requested = transform_spec(source, target, selection)
    limit = min(
        _torch_capabilities(source).triangular_lmax,
        _torch_capabilities(target).triangular_lmax,
    )

    if requested.lmax > limit and (source.kind == "cc" or target.kind == "cc"):
        raise ValueError(
            _cc_bandwidth_error(
                source,
                target,
                limit,
                requested_lmax=None if selection is None else selection.lmax,
            )
        )

    if requested.lmax != requested.mmax:
        raise ValueError(
            "torch-harmonics can only reproduce a triangular spharmgrid "
            "coefficient domain; truncation=None requests a non-triangular "
            "full bandwidth on these grids, so supply an explicit supported Tn"
        )
    if requested.lmax > limit:
        raise ValueError(
            f"requested lmax={requested.lmax} exceeds the verified "
            f"torch-harmonics triangular bandwidth T{limit} for these grids"
        )
    return TransformSpec(requested.lmax, requested.lmax)


def _cc_bandwidth_error(
    source: Grid,
    target: Grid,
    limit: int,
    *,
    requested_lmax: int | None,
) -> str:
    """Describe the current torch-harmonics CC bandwidth boundary."""
    cc_grids = [grid for grid in (source, target) if grid.kind == "cc"]
    cc_limit = min(_torch_capabilities(grid).triangular_lmax for grid in cc_grids)
    if requested_lmax is None:
        request = "the requested full spharmgrid transform domain"
    else:
        request = f"the requested T{requested_lmax} domain"
    pair_limit = "" if cc_limit == limit else f"; this grid pair is limited to T{limit}"
    return (
        "Current spharmgrid.torch support has a torch-harmonics bandwidth "
        "limitation for the supplied CC grid: the verified CC triangular "
        "limit is "
        f"T{cc_limit} (n <= min((nlat - 1) // 2, (nlon - 1) // 2)){pair_limit}; "
        f"{request} cannot currently be represented. Explicitly truncated "
        "filter, regrid, and regrid_vector calls remain available within the "
        f"supported range through T{limit}."
    )


def _torch_grid_name(grid: Grid) -> str:
    return "legendre-gauss" if grid.kind == "gl" else "equiangular"


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
        vector: bool,
    ) -> None:
        super().__init__()
        if spec.lmax != spec.mmax:
            raise ValueError("torch-harmonics transform state must be triangular")

        self.source_grid = source
        self.target_grid = target
        self.spec = spec

        source_layout = grid_layout(source)
        target_layout = grid_layout(target)
        self.register_buffer(
            "_source_latitude_indices",
            torch.as_tensor(source_layout.latitude.canonical_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_source_longitude_indices",
            torch.as_tensor(
                source_layout.longitude.canonical_indices,
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_target_latitude_restore",
            torch.as_tensor(target_layout.latitude.restore_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_target_longitude_restore",
            torch.as_tensor(target_layout.longitude.restore_indices, dtype=torch.long),
            persistent=False,
        )

        modes = torch.arange(spec.mmax + 1, dtype=torch.float64)
        source_angle = modes * source_layout.phi0_radians
        target_angle = modes * target_layout.phi0_radians
        self.register_buffer(
            "_source_to_zero_phase_real",
            torch.cos(source_angle),
            persistent=False,
        )
        self.register_buffer(
            "_source_to_zero_phase_imag",
            -torch.sin(source_angle),
            persistent=False,
        )
        self.register_buffer(
            "_target_from_zero_phase_real",
            torch.cos(target_angle),
            persistent=False,
        )
        self.register_buffer(
            "_target_from_zero_phase_imag",
            torch.sin(target_angle),
            persistent=False,
        )
        self.register_buffer(
            "_degrees",
            torch.arange(spec.lmax + 1, dtype=torch.float64).reshape(-1, 1),
            persistent=False,
        )

        grid_name = _torch_grid_name(source)
        self._scalar_analysis = _torch_harmonics.RealSHT(
            source.nlat,
            source.nlon,
            lmax=spec.lmax + 1,
            mmax=spec.mmax + 1,
            grid=grid_name,
            norm="ortho",
            csphase=True,
        )
        self._scalar_synthesis = _torch_harmonics.InverseRealSHT(
            target.nlat,
            target.nlon,
            lmax=spec.lmax + 1,
            mmax=spec.mmax + 1,
            grid=_torch_grid_name(target),
            norm="ortho",
            csphase=True,
        )
        self._vector_analysis: Any = None
        self._vector_synthesis: Any = None
        if vector:
            self._vector_analysis = _torch_harmonics.RealVectorSHT(
                source.nlat,
                source.nlon,
                lmax=spec.lmax + 1,
                mmax=spec.mmax + 1,
                grid=grid_name,
                norm="ortho",
                csphase=True,
            )
            self._vector_synthesis = _torch_harmonics.InverseRealVectorSHT(
                target.nlat,
                target.nlon,
                lmax=spec.lmax + 1,
                mmax=spec.mmax + 1,
                grid=_torch_grid_name(target),
                norm="ortho",
                csphase=True,
            )

    def _move_to(self, device: torch.device) -> _TorchTransform:
        self.to(device=device)
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
            raise RuntimeError("vector transform state was not requested")
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
            raise RuntimeError("vector transform state was not requested")
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


def _spectral_weights(
    state: _TorchTransform,
    selection: SpectralRange,
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
    selection: SpectralRange,
    taper: float | None,
) -> torch.Tensor:
    weights = _spectral_weights(state, selection, taper, coefficients.real.dtype)
    return coefficients * weights


def _make_state(
    source: Grid,
    target: Grid,
    selection: SpectralRange | None,
    *,
    vector: bool,
    device: torch.device,
) -> _TorchTransform:
    spec = _resolve_transform_spec(source, target, selection)
    state = _TorchTransform(source, target, spec, vector=vector)
    return state._move_to(device)


def _check_selection(
    selection: SpectralRange | None,
    state: _TorchTransform,
) -> None:
    if selection is not None and selection.lmax > state.spec.lmax:
        raise ValueError(
            f"requested lmax={selection.lmax} exceeds transform lmax={state.spec.lmax}"
        )


def _require_vector_bandwidth(state: _TorchTransform) -> None:
    if state.spec.lmax < 1:
        raise ValueError("vector operation requires a grid supporting total degree l=1")
