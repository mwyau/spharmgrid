# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental public-API adapter for atmospheric GL longitude counts."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import s2fft
from jax import Array


def _downsample_longitude(field: Array, bandlimit: int) -> Array:
    """Fourier-resample a GL field from ``2L`` to S2FFT's ``2L-1`` nodes."""
    if field.shape[-1] != 2 * bandlimit:
        raise ValueError("the experimental public GL adapter requires 2L longitudes")

    # fftshift orders these bins as -L, ..., -1, 0, ..., L-1.  The first
    # slot is the even-length FFT Nyquist bin (m=-L, equivalent to m=+L on
    # these samples), which is outside S2FFT's |m| < L coefficient domain.
    spectrum = jnp.fft.fftshift(jnp.fft.fft(field, axis=-1, norm="forward"), axes=-1)
    spectrum = spectrum[..., 1:]

    # ``norm='forward'`` leaves the inverse FFT unscaled, so these normalized
    # Fourier coefficients evaluate the same band-limited series at 2L-1 nodes.
    return jnp.fft.ifft(jnp.fft.ifftshift(spectrum, axes=-1), axis=-1, norm="forward")


def _upsample_longitude(field: Array, bandlimit: int) -> Array:
    """Fourier-resample S2FFT's GL nodes onto the atmospheric ``2L`` nodes."""
    if field.shape[-1] != 2 * bandlimit - 1:
        raise ValueError("S2FFT GL synthesis must have 2L-1 longitudes")

    spectrum = jnp.fft.fftshift(jnp.fft.fft(field, axis=-1, norm="forward"), axes=-1)
    # Centered even-length order is -L, ..., -1, 0, ..., L-1.  Insert a zero
    # in its first slot; the 2L physical grid has no spherical harmonic m=-L.
    spectrum = jnp.concatenate((jnp.zeros_like(spectrum[..., :1]), spectrum), axis=-1)
    return jnp.fft.ifft(jnp.fft.ifftshift(spectrum, axes=-1), axis=-1, norm="forward")


def forward_public_resample(
    field: Array,
    bandlimit: int,
    spin: int,
    precomps: tuple[Array, ...],
) -> Array:
    """Analyze a ``(L, 2L)`` GL field through public S2FFT transforms."""
    native_field = _downsample_longitude(field, bandlimit)
    if spin == 0:
        native_field = jnp.real(native_field)
    return cast(
        Array,
        s2fft.forward_jax(
            native_field,
            bandlimit,
            spin=spin,
            sampling="gl",
            reality=spin == 0,
            precomps=precomps,
            spmd=False,
        ),
    )


def inverse_public_resample(
    coefficients: Array,
    bandlimit: int,
    spin: int,
    precomps: tuple[Array, ...],
) -> Array:
    """Synthesize a ``(L, 2L)`` GL field through public S2FFT transforms."""
    native_field = s2fft.inverse_jax(
        coefficients,
        bandlimit,
        spin=spin,
        sampling="gl",
        reality=spin == 0,
        precomps=precomps,
        spmd=False,
    )
    return _upsample_longitude(native_field, bandlimit)
