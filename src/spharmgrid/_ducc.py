# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small internal adapters around DUCC0's spherical-harmonic functions."""

from __future__ import annotations

from functools import cache, lru_cache
from numbers import Integral
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._transform import TransformSpec
from .grids import Grid

Geometry = Literal["CC", "GL"]

DEFAULT_DASK_SHT_THREADS = 1
_ALM_SUBSELECTION_CACHE_SIZE = 32


def resolve_sht_threads(sht_threads: int | None, *, dask: bool) -> int:
    """Resolve and validate a public per-operation DUCC thread setting.

    An eager operation leaves thread selection to DUCC when no explicit value
    is supplied.  Dask-backed operations use one DUCC thread per transform
    unless the caller supplies a value, leaving task-level concurrency to the
    caller.
    """
    if sht_threads is None:
        return DEFAULT_DASK_SHT_THREADS if dask else 0
    if isinstance(sht_threads, bool) or not isinstance(sht_threads, Integral):
        raise TypeError("sht_threads must be a positive integer or None")
    resolved = int(sht_threads)
    if resolved <= 0:
        raise ValueError("sht_threads must be a positive integer or None")
    return resolved


def geometry_for(grid: Grid) -> Geometry:
    """Map the public lowercase grid label to DUCC's geometry label."""
    return "GL" if grid.kind == "gl" else "CC"


@cache
def alm_degrees(lmax: int, mmax: int) -> NDArray[np.int64]:
    """Return total degree for each contiguous healpy-ordered coefficient."""
    degrees = np.concatenate(
        [np.arange(m, lmax + 1, dtype=np.int64) for m in range(mmax + 1)]
    )
    degrees.setflags(write=False)
    return degrees


@cache
def alm_orders(lmax: int, mmax: int) -> NDArray[np.int64]:
    """Return zonal order for each contiguous healpy-ordered coefficient."""
    orders = np.concatenate(
        [np.full(lmax - m + 1, m, dtype=np.int64) for m in range(mmax + 1)]
    )
    orders.setflags(write=False)
    return orders


@lru_cache(maxsize=_ALM_SUBSELECTION_CACHE_SIZE)
def alm_subselection(
    source_lmax: int,
    source_mmax: int,
    target_lmax: int,
    target_mmax: int,
) -> NDArray[np.intp]:
    """Return packed source positions for a target DUCC coefficient domain.

    DUCC stores coefficients in contiguous order blocks for each non-negative
    zonal order.  A target domain used for synthesis is therefore a subset of
    a source domain whenever its degree and order limits are no larger.
    """
    if target_lmax > source_lmax or target_mmax > source_mmax:
        raise ValueError("target coefficient domain exceeds the source domain")

    target_size = (target_mmax + 1) * (target_lmax + 1) - (
        target_mmax * (target_mmax + 1) // 2
    )
    result = np.empty(target_size, dtype=np.intp)
    destination = 0
    for order in range(target_mmax + 1):
        block_size = target_lmax - order + 1
        source_offset = order * (source_lmax + 1) - order * (order - 1) // 2
        result[destination : destination + block_size] = source_offset + np.arange(
            block_size, dtype=np.intp
        )
        destination += block_size
    result.setflags(write=False)
    return result


def scalar_analysis(
    frame: NDArray[np.generic],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    phi0: float,
    nthreads: int,
) -> NDArray[np.complex128]:
    """Analyze one north-to-south, cyclic-eastward scalar map."""
    _require_frame(frame)
    import ducc0

    values = np.ascontiguousarray(frame, dtype=np.float64)
    result = ducc0.sht.analysis_2d(
        map=values[np.newaxis, :, :],
        spin=0,
        lmax=spec.lmax,
        mmax=spec.mmax,
        geometry=geometry,
        phi0=phi0,
        nthreads=nthreads,
    )
    return cast(NDArray[np.complex128], result)


def scalar_synthesis(
    alm: NDArray[np.complexfloating],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    ntheta: int,
    nphi: int,
    phi0: float,
    nthreads: int,
) -> NDArray[np.float64]:
    """Synthesize one scalar map on a north-to-south DUCC geometry."""
    import ducc0

    values = np.ascontiguousarray(alm)
    result = ducc0.sht.synthesis_2d(
        alm=values,
        spin=0,
        lmax=spec.lmax,
        mmax=spec.mmax,
        geometry=geometry,
        ntheta=ntheta,
        nphi=nphi,
        phi0=phi0,
        nthreads=nthreads,
    )
    return cast(NDArray[np.float64], result[0])


def scalar_derivative_synthesis(
    alm: NDArray[np.complexfloating],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    ntheta: int,
    nphi: int,
    phi0: float,
    nthreads: int,
) -> NDArray[np.float64]:
    """Synthesize theta and eastward angular derivatives of a scalar field.

    DUCC's ``DERIV1`` mode returns ``(d/dtheta, (1/sin(theta)) d/dphi)``.
    The caller maps the first component from southward theta to northward
    latitude and applies the physical-radius factor.
    """
    import ducc0

    result = ducc0.sht.synthesis_2d(
        alm=np.ascontiguousarray(alm),
        spin=1,
        lmax=spec.lmax,
        mmax=spec.mmax,
        geometry=geometry,
        ntheta=ntheta,
        nphi=nphi,
        phi0=phi0,
        nthreads=nthreads,
        mode="DERIV1",
    )
    return cast(NDArray[np.float64], result)


def vector_analysis(
    u: NDArray[np.generic],
    v: NDArray[np.generic],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    phi0: float,
    nthreads: int,
) -> NDArray[np.complex128]:
    """Analyze geographic eastward/northward wind into DUCC E/B coefficients.

    DUCC's spin-1 map components are ``(v_theta, v_phi)``.  Geographic wind
    uses northward ``v``, while increasing theta points south, so spharmgrid
    maps the components as ``(-v, u)``.
    """
    _require_frame(u)
    _require_frame(v)
    if u.shape != v.shape:
        raise ValueError(
            f"wind frames have incompatible shapes: {u.shape} and {v.shape}"
        )
    import ducc0

    eastward = np.asarray(u, dtype=np.float64)
    northward = np.asarray(v, dtype=np.float64)
    vector_map = np.stack((-northward, eastward), axis=0)
    result = ducc0.sht.analysis_2d(
        map=np.ascontiguousarray(vector_map),
        spin=1,
        lmax=spec.lmax,
        mmax=spec.mmax,
        geometry=geometry,
        phi0=phi0,
        nthreads=nthreads,
    )
    return cast(NDArray[np.complex128], result)


def vector_synthesis(
    alm_e: NDArray[np.complexfloating],
    alm_b: NDArray[np.complexfloating],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    ntheta: int,
    nphi: int,
    phi0: float,
    nthreads: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Synthesize DUCC E/B coefficients into geographic eastward/northward wind."""
    import ducc0

    coefficients = np.stack((alm_e, alm_b), axis=0)
    result = ducc0.sht.synthesis_2d(
        alm=np.ascontiguousarray(coefficients),
        spin=1,
        lmax=spec.lmax,
        mmax=spec.mmax,
        geometry=geometry,
        ntheta=ntheta,
        nphi=nphi,
        phi0=phi0,
        nthreads=nthreads,
    )
    # This is the inverse of ``(-v, u)`` in ``vector_analysis``.
    return (
        cast(NDArray[np.float64], result[1]),
        cast(NDArray[np.float64], -result[0]),
    )


def _require_frame(frame: NDArray[np.generic]) -> None:
    if frame.ndim != 2:
        raise ValueError(
            f"DUCC kernel requires a two-dimensional frame, got {frame.shape}"
        )
