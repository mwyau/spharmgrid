# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark-only routing for the two validated GL (L, 2L) adapters."""

from __future__ import annotations

from _candidate_a import forward_public_resample, inverse_public_resample
from jax import Array

import spharmgrid.jax._backend as _backend

METHODS = ("public_resample", "internal_ftm")
_selected_method = "internal_ftm"
_forward_one = _backend._forward_one
_inverse_one = _backend._inverse_one


def select_method(method: str) -> None:
    """Select a comparison adapter for the benchmark process."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}")
    global _selected_method
    _selected_method = method


def _forward_with_candidate(
    field: Array,
    bandlimit: int,
    sampling: str,
    spin: int,
) -> Array:
    if (
        _selected_method == "public_resample"
        and sampling == "gl"
        and field.shape[-1] == 2 * bandlimit
    ):
        precomps = _backend._precomputes(bandlimit, sampling, spin, forward=True)
        return forward_public_resample(field, bandlimit, spin, precomps)
    return _forward_one(field, bandlimit, sampling, spin)


def _inverse_with_candidate(
    coefficients: Array,
    bandlimit: int,
    sampling: str,
    spin: int,
    nlon: int,
) -> Array:
    if (
        _selected_method == "public_resample"
        and sampling == "gl"
        and nlon == 2 * bandlimit
    ):
        precomps = _backend._precomputes(bandlimit, sampling, spin, forward=False)
        return inverse_public_resample(coefficients, bandlimit, spin, precomps)
    return _inverse_one(coefficients, bandlimit, sampling, spin, nlon)


# This selector is imported by the benchmark and validation scripts only.
_backend._forward_one = _forward_with_candidate
_backend._inverse_one = _inverse_with_candidate
