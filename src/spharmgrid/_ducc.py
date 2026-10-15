# SPDX-FileCopyrightText: 2026 Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small internal adapters around DUCC0's spherical-harmonic functions."""

from __future__ import annotations

from functools import lru_cache
from math import prod
from numbers import Integral
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._transform import TransformSpec
from .grids import Grid

Geometry = Literal["CC", "F1", "GL"]

DEFAULT_DASK_SHT_THREADS = 1
_ALM_INDEX_CACHE_SIZE = 8
_ALM_INDEX_CACHE_MAX_BYTES = 2 * 1024 * 1024


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
    return cast(Geometry, grid.kind.upper())


def _alm_size(lmax: int, mmax: int) -> int:
    """Return the number of packed coefficients for inclusive degree/order limits."""
    return (mmax + 1) * (lmax + 1) - mmax * (mmax + 1) // 2


def _build_alm_degrees(lmax: int, mmax: int) -> NDArray[np.int64]:
    degrees = np.empty(_alm_size(lmax, mmax), dtype=np.int64)
    destination = 0
    for order in range(mmax + 1):
        block_size = lmax - order + 1
        degrees[destination : destination + block_size] = np.arange(
            order, lmax + 1, dtype=np.int64
        )
        destination += block_size
    degrees.setflags(write=False)
    return degrees


def _build_alm_orders(lmax: int, mmax: int) -> NDArray[np.int64]:
    orders = np.empty(_alm_size(lmax, mmax), dtype=np.int64)
    destination = 0
    for order in range(mmax + 1):
        block_size = lmax - order + 1
        orders[destination : destination + block_size] = order
        destination += block_size
    orders.setflags(write=False)
    return orders


@lru_cache(maxsize=_ALM_INDEX_CACHE_SIZE)
def _cached_alm_degrees(lmax: int, mmax: int) -> NDArray[np.int64]:
    return _build_alm_degrees(lmax, mmax)


@lru_cache(maxsize=_ALM_INDEX_CACHE_SIZE)
def _cached_alm_orders(lmax: int, mmax: int) -> NDArray[np.int64]:
    return _build_alm_orders(lmax, mmax)


def _cache_alm_indices(lmax: int, mmax: int) -> bool:
    size_bytes = _alm_size(lmax, mmax) * np.dtype(np.int64).itemsize
    return size_bytes <= _ALM_INDEX_CACHE_MAX_BYTES


def alm_degrees(lmax: int, mmax: int) -> NDArray[np.int64]:
    """Return total degree for each contiguous healpy-ordered coefficient."""
    if _cache_alm_indices(lmax, mmax):
        return _cached_alm_degrees(lmax, mmax)
    return _build_alm_degrees(lmax, mmax)


def alm_orders(lmax: int, mmax: int) -> NDArray[np.int64]:
    """Return zonal order for each contiguous healpy-ordered coefficient."""
    if _cache_alm_indices(lmax, mmax):
        return _cached_alm_orders(lmax, mmax)
    return _build_alm_orders(lmax, mmax)


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

    target_size = _alm_size(target_lmax, target_mmax)
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
    field: NDArray[np.generic],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    phi0: float,
    nthreads: int,
) -> NDArray[np.complex128]:
    """Analyze scalar maps with shape ``(..., ntheta, nphi)``."""
    import ducc0

    leading_shape, frames = _flatten_map_batch(field)
    mode_size = _alm_size(spec.lmax, spec.mmax)
    result = np.empty((frames.shape[0], 1, mode_size), dtype=np.complex128)
    for index in range(frames.shape[0]):
        frame = frames[index]
        coefficients = ducc0.sht.analysis_2d(
            map=frame[np.newaxis, :, :],
            spin=0,
            lmax=spec.lmax,
            mmax=spec.mmax,
            geometry=geometry,
            phi0=phi0,
            nthreads=nthreads,
        )
        result[index] = coefficients
    return result.reshape(leading_shape + (1, mode_size))


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
    """Synthesize scalar coefficient batches on a DUCC geometry."""
    import ducc0

    leading_shape, frames = _flatten_scalar_coefficients(alm)
    result = np.empty((frames.shape[0], ntheta, nphi), dtype=np.float64)
    for index, coefficients in enumerate(frames):
        values = ducc0.sht.synthesis_2d(
            alm=coefficients,
            spin=0,
            lmax=spec.lmax,
            mmax=spec.mmax,
            geometry=geometry,
            ntheta=ntheta,
            nphi=nphi,
            phi0=phi0,
            nthreads=nthreads,
        )
        result[index] = values[0]
    return result.reshape(leading_shape + (ntheta, nphi))


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
    """Synthesize derivatives from scalar coefficient batches.

    DUCC's ``DERIV1`` mode returns ``(d/dtheta, (1/sin(theta)) d/dphi)``.
    The caller maps the first component from southward theta to northward
    latitude and applies the physical-radius factor. The result has shape
    ``(..., 2, ntheta, nphi)``.
    """
    import ducc0

    leading_shape, frames = _flatten_scalar_coefficients(alm)
    result = np.empty((frames.shape[0], 2, ntheta, nphi), dtype=np.float64)
    for index, coefficients in enumerate(frames):
        values = ducc0.sht.synthesis_2d(
            alm=coefficients,
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
        result[index] = values
    return result.reshape(leading_shape + (2, ntheta, nphi))


def vector_analysis(
    u: NDArray[np.generic],
    v: NDArray[np.generic],
    *,
    spec: TransformSpec,
    geometry: Geometry,
    phi0: float,
    nthreads: int,
) -> NDArray[np.complex128]:
    """Analyze batched geographic wind into DUCC E/B coefficients.

    DUCC's spin-1 map components are ``(v_theta, v_phi)``.  Geographic wind
    uses northward ``v``, while increasing theta points south, so spharmgrid
    maps the components as ``(-v, u)``. Inputs have shape
    ``(..., ntheta, nphi)`` and output has shape ``(..., 2, nalm)``.
    """
    leading_shape, eastward_frames = _flatten_map_batch(u)
    other_leading_shape, northward_frames = _flatten_map_batch(v)
    if leading_shape != other_leading_shape:
        raise ValueError(
            f"wind batches have incompatible leading shapes: "
            f"{leading_shape} and {other_leading_shape}"
        )
    if u.shape != v.shape:
        raise ValueError(
            f"wind batches have incompatible shapes: {u.shape} and {v.shape}"
        )
    import ducc0

    mode_size = _alm_size(spec.lmax, spec.mmax)
    result = np.empty((eastward_frames.shape[0], 2, mode_size), dtype=np.complex128)
    for index in range(eastward_frames.shape[0]):
        eastward = eastward_frames[index]
        northward = northward_frames[index]
        vector_map = np.stack((-northward, eastward), axis=0)
        coefficients = ducc0.sht.analysis_2d(
            map=np.ascontiguousarray(vector_map),
            spin=1,
            lmax=spec.lmax,
            mmax=spec.mmax,
            geometry=geometry,
            phi0=phi0,
            nthreads=nthreads,
        )
        result[index] = coefficients
    return result.reshape(leading_shape + (2, mode_size))


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
    """Synthesize DUCC E/B coefficient batches into geographic wind."""
    import ducc0

    if alm_e.shape != alm_b.shape or alm_e.ndim < 1:
        raise ValueError(
            "eastward and northward wind coefficients must have matching "
            f"shapes, got {alm_e.shape} and {alm_b.shape}"
        )
    leading_shape = alm_e.shape[:-1]
    batch_size = prod(leading_shape) if leading_shape else 1
    electric = np.ascontiguousarray(alm_e).reshape(batch_size, alm_e.shape[-1])
    magnetic = np.ascontiguousarray(alm_b).reshape(batch_size, alm_b.shape[-1])
    eastward = np.empty((batch_size, ntheta, nphi), dtype=np.float64)
    northward = np.empty_like(eastward)
    for index in range(batch_size):
        e_coefficients = electric[index]
        b_coefficients = magnetic[index]
        coefficients = np.stack((e_coefficients, b_coefficients), axis=0)
        values = ducc0.sht.synthesis_2d(
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
        eastward[index] = values[1]
        northward[index] = -values[0]
    shape = leading_shape + (ntheta, nphi)
    return (
        cast(NDArray[np.float64], eastward.reshape(shape)),
        cast(NDArray[np.float64], northward.reshape(shape)),
    )


def _flatten_map_batch(
    field: NDArray[np.generic],
) -> tuple[tuple[int, ...], NDArray[np.float64]]:
    """Flatten leading map axes while retaining the input leading shape."""
    values = np.asarray(field)
    if values.ndim < 2:
        raise ValueError(
            f"DUCC maps must have at least two dimensions, got {values.shape}"
        )
    leading_shape = values.shape[:-2]
    batch_size = prod(leading_shape) if leading_shape else 1
    frames = np.ascontiguousarray(
        values.reshape((batch_size,) + values.shape[-2:]), dtype=np.float64
    )
    return leading_shape, frames


def _flatten_scalar_coefficients(
    alm: NDArray[np.complexfloating],
) -> tuple[tuple[int, ...], NDArray[np.complex128]]:
    """Flatten scalar coefficient batches with the DUCC component axis intact."""
    values = np.asarray(alm)
    if values.ndim < 2 or values.shape[-2] != 1:
        raise ValueError(
            "scalar DUCC coefficients must have shape (..., 1, nalm), "
            f"got {values.shape}"
        )
    leading_shape = values.shape[:-2]
    batch_size = prod(leading_shape) if leading_shape else 1
    frames = np.ascontiguousarray(values, dtype=np.complex128).reshape(
        batch_size, 1, values.shape[-1]
    )
    return leading_shape, frames
