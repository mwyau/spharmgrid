# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map regular GL longitude FFTs into S2FFT's fixed GL Fourier domain.

S2FFT 1.4.0 exposes GL sampling with ``(L, 2L - 1)`` physical points. A common
regular atmospheric GL grid uses ``(L, 2L)``, but spharmgrid's GL grid permits
any valid longitude count. The spherical harmonic basis still contains only
``2L - 1`` orders. spharmgrid owns the length-``nlon`` longitude FFT, omits
unrepresentable modes and an even-length Nyquist bin, and inserts zero bins on
synthesis; S2FFT performs the latitudinal transform. This internal boundary is
tracked at https://github.com/astro-informatics/s2fft/issues/406 and should be
revisited when S2FFT exposes a suitable public API.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import cast

import jax.numpy as jnp
from jax import Array

_S2FFT_VERSION = "1.4.0"
_S2FFT_TAG_COMMIT = "e140536880fc53081a98156fb0245d9d197d1e5d"
_GL_SUPPORT = (
    "spharmgrid's regular GL JAX support currently depends on S2FFT 1.4.0 "
    "internals; another version is unsupported until verified. This restriction "
    "can be relaxed once S2FFT provides a suitable public API."
)
_FORWARD_PARAMETERS = (
    "ftm_in",
    "beta_in",
    "L",
    "spin",
    "nside",
    "sampling",
    "reality",
    "precomps",
    "spmd",
    "L_lower",
)
_INVERSE_PARAMETERS = (
    "flm",
    "beta",
    "L",
    "spin",
    "nside",
    "sampling",
    "reality",
    "precomps",
    "spmd",
    "L_lower",
)


@lru_cache(maxsize=1)
def _internal_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load and validate the pinned internal S2FFT GL latitude interface."""
    try:
        installed_version = version("s2fft")
    except PackageNotFoundError as error:
        raise RuntimeError(f"{_GL_SUPPORT} S2FFT is not installed.") from error
    if installed_version != _S2FFT_VERSION:
        raise RuntimeError(
            f"{_GL_SUPPORT} Found s2fft=={installed_version}; the validated "
            f"release is s2fft=={_S2FFT_VERSION} (tag commit {_S2FFT_TAG_COMMIT})."
        )

    try:
        otf = importlib.import_module("s2fft.transforms.otf_recursions")
        samples = importlib.import_module("s2fft.sampling.s2_samples")
        quadrature = importlib.import_module("s2fft.utils.quadrature_jax")
        forward = otf.forward_latitudinal_step_jax
        inverse = otf.inverse_latitudinal_step_jax
        if not callable(samples.thetas) or not callable(
            quadrature.quad_weights_transform
        ):
            raise AttributeError("missing GL sampling or quadrature helpers")
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            f"{_GL_SUPPORT} S2FFT {installed_version} does not expose the "
            "required GL latitude-step interface."
        ) from error

    for function, expected in (
        (forward, _FORWARD_PARAMETERS),
        (inverse, _INVERSE_PARAMETERS),
    ):
        actual = tuple(inspect.signature(function).parameters)
        if actual[: len(expected)] != expected:
            raise RuntimeError(
                "the s2fft==1.4.0 internal GL latitude-step signature changed; "
                "verify the required symbols and signatures before changing the pin"
            )
    return otf, samples, quadrature


def _retained_order_count(bandlimit: int, nlon: int) -> int:
    """Return the largest physical zonal order representable by the grid."""
    if nlon < 2:
        raise ValueError("regular GL longitude count must be at least 2")
    return min(bandlimit - 1, (nlon - 1) // 2)


def _centered_slice(center: int, half_width: int) -> slice:
    return slice(center - half_width, center + half_width + 1)


def forward_ftm_to_flm(
    field: Array,
    bandlimit: int,
    spin: int,
    precomps: tuple[Array, ...],
) -> Array:
    """Analyze one ``(L, nlon)`` regular GL field through S2FFT's ``ftm``."""
    if field.ndim != 2 or field.shape[0] != bandlimit:
        raise ValueError("the regular GL transform requires one (L, nlon) frame")
    nlon = field.shape[-1]
    m_keep = _retained_order_count(bandlimit, nlon)
    otf, samples, quadrature = _internal_modules()
    reality = spin == 0
    fixed_center = bandlimit - 1

    if reality:
        positive = jnp.fft.rfft(jnp.real(field), axis=-1, norm="forward")
        positive = positive[..., : m_keep + 1]
        ftm = (
            jnp.zeros((bandlimit, 2 * bandlimit - 1), dtype=positive.dtype)
            .at[:, fixed_center : fixed_center + m_keep + 1]
            .set(positive * (2 * bandlimit - 1))
        )
    else:
        physical_center = nlon // 2
        spectrum = jnp.fft.fftshift(
            jnp.fft.fft(field, axis=-1, norm="forward"), axes=-1
        )
        physical_slice = _centered_slice(physical_center, m_keep)
        fixed_slice = _centered_slice(fixed_center, m_keep)
        retained = spectrum[:, physical_slice]
        ftm = (
            jnp.zeros((bandlimit, 2 * bandlimit - 1), dtype=retained.dtype)
            .at[:, fixed_slice]
            .set(retained * (2 * bandlimit - 1))
        )

    # S2FFT's latitude primitive expects the unnormalized FFT on its fixed
    # width-2L-1 GL domain, followed by its theta quadrature weights.
    ftm = ftm * quadrature.quad_weights_transform(bandlimit, "gl", None)[:, None]
    thetas = jnp.asarray(samples.thetas(bandlimit, "gl"), dtype=jnp.float64)
    flm = otf.forward_latitudinal_step_jax(
        ftm,
        thetas,
        bandlimit,
        spin,
        None,
        "gl",
        reality,
        precomps=precomps,
        spmd=False,
        L_lower=0,
    )

    # Match convention steps in s2fft.transforms.spherical.forward_jax.
    degrees = jnp.arange(bandlimit, dtype=jnp.float64)
    flm = flm * jnp.sqrt((2.0 * degrees + 1.0) / (4.0 * jnp.pi))[:, None]
    if reality:
        flm = flm.at[:, :fixed_center].set(
            jnp.flip(
                (-1) ** (jnp.arange(1, bandlimit) % 2) * jnp.conj(flm[:, bandlimit:]),
                axis=-1,
            )
        )
    flm = jnp.where((degrees < abs(spin))[:, None], jnp.zeros_like(flm), flm)
    orders = jnp.abs(jnp.arange(-fixed_center, fixed_center + 1))
    flm = jnp.where(orders[None, :] <= m_keep, flm, jnp.zeros_like(flm))
    return cast(Array, flm * (-1) ** abs(spin))


def inverse_flm_to_ftm(
    coefficients: Array,
    bandlimit: int,
    spin: int,
    nlon: int,
    precomps: tuple[Array, ...],
) -> Array:
    """Synthesize one ``(L, nlon)`` field through S2FFT's fixed-width ``ftm``."""
    if coefficients.shape != (bandlimit, 2 * bandlimit - 1):
        raise ValueError(
            "the regular GL inverse requires (L, 2L-1) spherical harmonic coefficients"
        )
    m_keep = _retained_order_count(bandlimit, nlon)
    otf, samples, _ = _internal_modules()
    degrees = jnp.arange(bandlimit, dtype=jnp.float64)
    normalized = (
        coefficients * jnp.sqrt((2.0 * degrees + 1.0) / (4.0 * jnp.pi))[:, None]
    )
    thetas = jnp.asarray(samples.thetas(bandlimit, "gl"), dtype=jnp.float64)
    reality = spin == 0
    ftm = otf.inverse_latitudinal_step_jax(
        normalized,
        thetas,
        bandlimit,
        spin,
        None,
        "gl",
        reality,
        precomps=precomps,
        spmd=False,
        L_lower=0,
    )
    ftm = ftm * (-1) ** abs(spin)
    fixed_center = bandlimit - 1
    if reality:
        ftm = ftm.at[:, :fixed_center].set(
            jnp.flip(jnp.conj(ftm[:, bandlimit:]), axis=-1)
        )

    physical_center = nlon // 2
    fixed_slice = _centered_slice(fixed_center, m_keep)
    physical_slice = _centered_slice(physical_center, m_keep)
    modes = ftm[:, fixed_slice]
    # Initializing the FFT bins to zero excludes an even-length Nyquist bin.
    spectrum = (
        jnp.zeros((bandlimit, nlon), dtype=ftm.dtype).at[:, physical_slice].set(modes)
    )
    field = jnp.fft.ifft(jnp.fft.ifftshift(spectrum, axes=-1), axis=-1, norm="forward")
    return jnp.real(field) if spin == 0 else field
