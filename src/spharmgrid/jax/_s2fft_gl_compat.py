# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge atmospheric regular GL longitude counts to S2FFT's GL latitude steps.

S2FFT 1.4.0 exposes GL ``(L, 2L - 1)`` sampling. Atmospheric regular Gaussian
grids commonly use ``(L, 2L)``, while the spherical harmonic basis still has
only ``2L - 1`` representable orders. spharmgrid computes the length-``2L``
longitude FFT, removes the aliased Nyquist mode, and passes the remaining
Fourier modes to S2FFT's latitude transform. Synthesis inserts a zero Nyquist
mode before the length-``2L`` inverse FFT. S2FFT continues to compute the
latitudinal transform.

This boundary uses S2FFT 1.4.0 internals while public support is tracked at
https://github.com/astro-informatics/s2fft/issues/406. Revisit and remove this
code when S2FFT exposes a public transform for the required grid.
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
_GL_2L_SUPPORT = (
    "spharmgrid's (L, 2L) JAX GL support currently depends on S2FFT 1.4.0 "
    "internals; other versions are unsupported until verified. This restriction "
    "can be relaxed once S2FFT exposes a suitable public API for this grid."
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
    """Load and validate the pinned S2FFT functions used for GL ``(L, 2L)``."""
    try:
        installed_version = version("s2fft")
    except PackageNotFoundError as error:
        raise RuntimeError(f"{_GL_2L_SUPPORT} S2FFT is not installed.") from error
    if installed_version != _S2FFT_VERSION:
        raise RuntimeError(
            f"{_GL_2L_SUPPORT} Found s2fft=={installed_version}; the validated "
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
            f"{_GL_2L_SUPPORT} S2FFT {installed_version} does not expose the "
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


def forward_ftm_to_flm(
    field: Array,
    bandlimit: int,
    spin: int,
    precomps: tuple[Array, ...],
) -> Array:
    """Analyze a ``(L, 2L)`` field after one longitude FFT."""
    if field.shape != (bandlimit, 2 * bandlimit):
        raise ValueError("the GL (L, 2L) transform requires a single (L, 2L) frame")

    otf, samples, quadrature = _internal_modules()
    reality = spin == 0
    if reality:
        # The positive rFFT endpoint is the length-2L Nyquist bin m=+L.
        # Remove it, then place m=0,...,L-1 in the full centered domain so
        # S2FFT's real-field latitude primitive can reconstruct negative m.
        spectrum = jnp.fft.rfft(jnp.real(field), axis=-1, norm="forward")[..., :-1]
        ftm = (
            jnp.zeros((bandlimit, 2 * bandlimit - 1), dtype=jnp.complex128)
            .at[:, bandlimit - 1 :]
            .set(spectrum * (2 * bandlimit - 1))
        )
    else:
        # Full complex fftshift order is -L,...,-1,0,...,L-1.  Its first
        # slot is the aliased Nyquist mode (-L == +L) and is removed.
        spectrum = jnp.fft.fftshift(
            jnp.fft.fft(field, axis=-1, norm="forward"), axes=-1
        )[..., 1:]
        ftm = spectrum * (2 * bandlimit - 1)

    # S2FFT's GL latitude primitive expects the unnormalized FFT used by its
    # public transform on 2L-1 nodes, followed by its theta quadrature weights.
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

    # These two convention steps mirror s2fft.transforms.spherical.forward_jax:
    # normalize spherical harmonics and zero degrees below the absolute spin.
    degrees = jnp.arange(bandlimit, dtype=jnp.float64)
    flm = flm * jnp.sqrt((2.0 * degrees + 1.0) / (4.0 * jnp.pi))[:, None]
    if reality:
        flm = flm.at[:, : bandlimit - 1].set(
            jnp.flip(
                (-1) ** (jnp.arange(1, bandlimit) % 2) * jnp.conj(flm[:, bandlimit:]),
                axis=-1,
            )
        )
    flm = jnp.where((degrees < abs(spin))[:, None], jnp.zeros_like(flm), flm)
    return cast(Array, flm * (-1) ** abs(spin))


def inverse_flm_to_ftm(
    coefficients: Array,
    bandlimit: int,
    spin: int,
    precomps: tuple[Array, ...],
) -> Array:
    """Synthesize a ``(L, 2L)`` field after one S2FFT latitude step."""
    if coefficients.shape != (bandlimit, 2 * bandlimit - 1):
        raise ValueError(
            "the GL (L, 2L) inverse requires (L, 2L-1) spherical harmonic coefficients"
        )

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
    if reality:
        ftm = ftm.at[:, : bandlimit - 1].set(
            jnp.flip(jnp.conj(ftm[:, bandlimit:]), axis=-1)
        )

    # Prepend the zero Nyquist slot in centered length-2L indexing.
    spectrum = jnp.concatenate((jnp.zeros_like(ftm[..., :1]), ftm), axis=-1)
    field = jnp.fft.ifft(jnp.fft.ifftshift(spectrum, axes=-1), axis=-1, norm="forward")
    return jnp.real(field) if spin == 0 else field
