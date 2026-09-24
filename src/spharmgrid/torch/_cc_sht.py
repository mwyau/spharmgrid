# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward analysis above the native CC quadrature limit."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch_harmonics as _torch_harmonics
from torch import Tensor
from torch_harmonics.quadrature import clenshaw_curtiss_weights


def _periodic_latitude_extension(values: Tensor, parity: Tensor) -> Tensor:
    """Continue pole-including latitude samples periodically."""
    parity_view = _mode_view(parity, values)
    return torch.cat((values, parity_view * values[..., 1:-1].flip(-1)), dim=-1)


def _periodic_latitude_extension_adjoint(
    values: Tensor,
    parity: Tensor,
    nlat: int,
) -> Tensor:
    """Fold a periodically extended latitude vector onto the original grid."""
    parity_view = _mode_view(parity, values)
    result = values[..., :nlat].clone()
    result[..., 1:-1] += parity_view * values[..., nlat:].flip(-1)
    return result


def _fourier_shift_latitude(values: Tensor, phase: Tensor) -> Tensor:
    """Shift periodic latitude samples by half a grid interval."""
    return cast(
        Tensor,
        torch.fft.ifft(
            torch.fft.fft(values, dim=-1) * phase,
            dim=-1,
        ),
    )


def _fourier_shift_latitude_adjoint(values: Tensor, phase: Tensor) -> Tensor:
    """Apply the adjoint of the periodic half-grid shift."""
    return _fourier_shift_latitude(values, phase.conj())


def _fold_resampled_latitude(
    values: Tensor,
    parity: Tensor,
    phase: Tensor,
    original_weights: Tensor,
    midpoint_weights: Tensor,
) -> Tensor:
    """Apply the folded midpoint quadrature correction to latitude modes."""
    extended = _periodic_latitude_extension(values, parity)
    shifted = _fourier_shift_latitude(extended, phase)
    weighted = shifted * midpoint_weights
    folded = _periodic_latitude_extension_adjoint(
        _fourier_shift_latitude_adjoint(weighted, phase),
        parity,
        values.shape[-1],
    )
    return original_weights * values + folded


def _mode_view(values: Tensor, reference: Tensor) -> Tensor:
    return values.to(device=reference.device, dtype=reference.real.dtype).reshape(
        *([1] * (reference.ndim - 2)),
        reference.shape[-2],
        1,
    )


def _native_projection_weights(
    nlat: int,
    nlon: int,
    lmax: int,
    mmax: int,
    *,
    vector: bool,
) -> Tensor:
    """Remove ordinary CC quadrature from native forward projection weights."""
    module_type = _torch_harmonics.RealVectorSHT if vector else _torch_harmonics.RealSHT
    native = module_type(
        nlat,
        nlon,
        lmax=lmax,
        mmax=mmax,
        grid="equiangular",
        norm="ortho",
        csphase=True,
    )
    native_lmax = getattr(native, "lmax", None)
    native_mmax = getattr(native, "mmax", None)
    if native_lmax != lmax or native_mmax != mmax:
        raise RuntimeError(
            "torch-harmonics resolved incompatible forward projection limits: "
            f"requested ({lmax}, {mmax}), got ({native_lmax}, {native_mmax})"
        )

    native_weights = getattr(native, "weights", None)
    if not isinstance(native_weights, torch.Tensor):
        raise RuntimeError(
            "torch-harmonics returned an incompatible supported forward-weight "
            "representation: expected a tensor in the native weights buffer"
        )
    expected_shape = (2, mmax, lmax, nlat) if vector else (mmax, lmax, nlat)
    if tuple(native_weights.shape) != expected_shape:
        raise RuntimeError(
            "torch-harmonics returned an incompatible supported forward-weight "
            f"representation: expected shape {expected_shape}, got "
            f"{tuple(native_weights.shape)}"
        )

    _, quadrature = clenshaw_curtiss_weights(nlat, -1.0, 1.0)
    quadrature = torch.as_tensor(quadrature)
    if quadrature.shape != (nlat,) or not torch.isfinite(quadrature).all():
        raise RuntimeError(
            "torch-harmonics returned incompatible ordinary CC quadrature: "
            f"expected finite shape {(nlat,)}, got {tuple(quadrature.shape)}"
        )
    if not torch.all(quadrature != 0):
        raise RuntimeError(
            "torch-harmonics returned zero ordinary CC quadrature weights"
        )

    quadrature = quadrature.to(device=native_weights.device, dtype=native_weights.dtype)
    # The module is temporary, so divide its buffer in place instead of holding
    # native quadrature weights and a second complete projection at once.
    native_weights.div_(quadrature.reshape((1,) * (native_weights.ndim - 1) + (nlat,)))
    return native_weights


def _cc_resampling_weights(nlat: int) -> tuple[Tensor, Tensor]:
    """Build quadrature weights for original and midpoint CC latitudes."""
    _, dense_quadrature = clenshaw_curtiss_weights(2 * nlat - 1, -1.0, 1.0)
    dense_quadrature = torch.as_tensor(dense_quadrature, dtype=torch.float64)
    if (
        dense_quadrature.shape != (2 * nlat - 1,)
        or not torch.isfinite(dense_quadrature).all()
        or not torch.all(dense_quadrature != 0)
    ):
        raise RuntimeError(
            "torch-harmonics returned invalid dense CC resampling quadrature"
        )
    original_weights = 2.0 * math.pi * dense_quadrature[::2]
    midpoint_values = 2.0 * math.pi * dense_quadrature[1::2]
    midpoint_weights = torch.cat(
        (midpoint_values, torch.zeros(nlat - 1, dtype=torch.float64))
    )
    if original_weights.shape != (nlat,) or midpoint_weights.shape != (2 * (nlat - 1),):
        raise RuntimeError("constructed CC quadrature weights have incompatible shapes")
    return original_weights, midpoint_weights


def _shift_phase(length: int) -> Tensor:
    frequencies = torch.fft.fftfreq(length, dtype=torch.float64)
    angle = math.pi * frequencies
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def _contract_complex(values: Tensor, weights: Tensor) -> Tensor:
    real = torch.einsum("...mk,mlk->...lm", values.real, weights)
    imaginary = torch.einsum("...mk,mlk->...lm", values.imag, weights)
    return torch.complex(real.contiguous(), imaginary.contiguous())


class _ExtendedCCSHT(torch.nn.Module):
    """Folded CC analysis using dense torch-harmonics projection weights."""

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: int,
        mmax: int,
        *,
        vector: bool,
    ) -> None:
        super().__init__()
        self.nlat = nlat
        self.nlon = nlon
        self.mmax = mmax
        self.register_buffer(
            "_projection",
            _native_projection_weights(
                nlat,
                nlon,
                lmax,
                mmax,
                vector=vector,
            ),
            persistent=False,
        )
        original_weights, midpoint_weights = _cc_resampling_weights(nlat)
        self.register_buffer("_original_weights", original_weights, persistent=False)
        self.register_buffer("_midpoint_weights", midpoint_weights, persistent=False)
        self.register_buffer(
            "_shift_phase",
            _shift_phase(2 * (nlat - 1)),
            persistent=False,
        )
        modes = torch.arange(mmax)
        parity = torch.where(modes.remainder(2) == 0, 1.0, -1.0)
        if vector:
            parity = -parity
        self.register_buffer("_parity", parity, persistent=False)

    def _resample(self, values: Tensor) -> Tensor:
        phase_values = cast(Tensor, self._shift_phase)
        phase = torch.complex(phase_values[:, 0], phase_values[:, 1])
        parity = cast(Tensor, self._parity)
        original_weights = cast(Tensor, self._original_weights)
        midpoint_weights = cast(Tensor, self._midpoint_weights)
        return _fold_resampled_latitude(
            values,
            parity,
            phase,
            original_weights,
            midpoint_weights,
        )


class _ExtendedCCRealSHT(_ExtendedCCSHT):
    """Scalar analysis above the native quadrature limit on a CC grid."""

    def __init__(self, nlat: int, nlon: int, *, lmax: int, mmax: int) -> None:
        super().__init__(nlat, nlon, lmax, mmax, vector=False)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim < 2:
            raise ValueError("CC scalar analysis expects at least two dimensions")
        if tuple(values.shape[-2:]) != (self.nlat, self.nlon):
            raise ValueError(
                "CC scalar analysis expects spatial dimensions "
                f"({self.nlat}, {self.nlon}), got {tuple(values.shape[-2:])}"
            )
        modes = torch.fft.rfft(values, dim=-1, norm="forward")[..., : self.mmax]
        weighted = self._resample(modes.transpose(-1, -2))
        projection = cast(Tensor, self._projection)
        return _contract_complex(weighted, projection)


class _ExtendedCCRealVectorSHT(_ExtendedCCSHT):
    """Vector analysis above the native quadrature limit on a CC grid."""

    def __init__(self, nlat: int, nlon: int, *, lmax: int, mmax: int) -> None:
        super().__init__(nlat, nlon, lmax, mmax, vector=True)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim < 3:
            raise ValueError("CC vector analysis expects at least three dimensions")
        if tuple(values.shape[-3:]) != (2, self.nlat, self.nlon):
            raise ValueError(
                "CC vector analysis expects spatial dimensions "
                f"(2, {self.nlat}, {self.nlon}), got {tuple(values.shape[-3:])}"
            )
        modes = torch.fft.rfft(values, dim=-1, norm="forward")[..., : self.mmax]
        modes = modes.transpose(-1, -2)
        component_zero = self._resample(modes[..., 0, :, :])
        component_one = self._resample(modes[..., 1, :, :])
        projection = cast(Tensor, self._projection)
        spheroidal = _contract_complex(component_zero, projection[0]) + 1j * (
            _contract_complex(component_one, projection[1])
        )
        toroidal = 1j * _contract_complex(component_zero, projection[1]) - (
            _contract_complex(component_one, projection[0])
        )
        return torch.stack((spheroidal, toroidal), dim=-3)
