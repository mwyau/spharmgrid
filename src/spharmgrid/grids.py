"""Supported global spherical grids and coordinate discovery.

The public grid descriptor is deliberately small.  ``ducc0`` performs the
actual spherical-harmonic transforms; this module validates the two sampling
geometries that spharmgrid passes to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

GridKind = Literal["gl", "cc"]

_LATITUDE_NAMES = ("lat", "latitude")
_LONGITUDE_NAMES = ("lon", "longitude")


@dataclass(frozen=True, slots=True)
class HorizontalCoordinates:
    """Names and dimensions of the horizontal coordinates on an xarray object."""

    latitude_name: str
    longitude_name: str
    latitude_dim: str
    longitude_dim: str


@dataclass(frozen=True, slots=True)
class AxisLayout:
    """A canonical DUCC axis order and the permutation back to user order."""

    canonical_indices: NDArray[np.intp]
    restore_indices: NDArray[np.intp]
    canonical_values: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GridLayout:
    """Coordinate permutations needed before and after a DUCC transform."""

    latitude: AxisLayout
    longitude: AxisLayout

    @property
    def phi0_radians(self) -> float:
        """Longitude of the canonical first column in radians."""
        return float(np.deg2rad(self.longitude.canonical_values[0]))


@dataclass(frozen=True, slots=True)
class GridCapabilities:
    """Transform limits documented by ``ducc0.sht.analysis_2d`` for a grid."""

    latitude_lmax: int
    longitude_mmax: int

    @property
    def triangular_lmax(self) -> int:
        """Largest fully triangular ``Tn`` transform supported by this grid."""
        return min(self.latitude_lmax, self.longitude_mmax)


@dataclass(frozen=True, slots=True)
class Grid:
    """A rectangular global Gauss--Legendre or Clenshaw--Curtis grid.

    Parameters
    ----------
    kind:
        ``"gl"`` for Gauss--Legendre nodes or ``"cc"`` for a pole-including
        Clenshaw--Curtis grid.
    latitude, longitude:
        One-dimensional coordinates in degrees.  Latitude may be ascending or
        descending.  Longitude must contain one complete, non-duplicated
        equally spaced cycle.
    """

    kind: GridKind
    latitude: NDArray[np.float64]
    longitude: NDArray[np.float64]

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind not in {"gl", "cc"}:
            raise ValueError("grid kind must be 'gl' or 'cc'")
        latitude = np.array(self.latitude, dtype=np.float64, copy=True)
        longitude = np.array(self.longitude, dtype=np.float64, copy=True)
        _validate_grid_arrays(kind, latitude, longitude)
        latitude.setflags(write=False)
        longitude.setflags(write=False)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)

    @property
    def nlat(self) -> int:
        """Number of latitude rings."""
        return int(self.latitude.size)

    @property
    def nlon(self) -> int:
        """Number of longitude points on each ring."""
        return int(self.longitude.size)


def gaussian_grid(
    nlat: int,
    nlon: int,
    *,
    lon0: float = 0.0,
    latitude_order: Literal["ascending", "descending"] = "ascending",
) -> Grid:
    """Construct a full rectangular Gauss--Legendre (GL) grid.

    ``ducc0.misc.GL_thetas`` supplies the Gaussian colatitudes.  Longitudes
    are equally spaced, east-positive degrees beginning at ``lon0``.
    """
    _validate_shape_arguments(nlat, nlon)
    _validate_latitude_order(latitude_order)
    import ducc0

    latitude = 90.0 - np.rad2deg(ducc0.misc.GL_thetas(nlat))
    if latitude_order == "ascending":
        latitude = latitude[::-1]
    longitude = _longitudes(nlon, lon0)
    return Grid("gl", latitude, longitude)


def clenshaw_curtis_grid(
    nlat: int,
    nlon: int,
    *,
    lon0: float = 0.0,
    latitude_order: Literal["ascending", "descending"] = "ascending",
) -> Grid:
    """Construct a pole-including Clenshaw--Curtis (CC) grid."""
    _validate_shape_arguments(nlat, nlon)
    _validate_latitude_order(latitude_order)
    latitude = np.linspace(-90.0, 90.0, nlat, dtype=np.float64)
    if latitude_order == "descending":
        latitude = latitude[::-1]
    longitude = _longitudes(nlon, lon0)
    return Grid("cc", latitude, longitude)


def detect_grid(field: xr.DataArray | xr.Dataset) -> Grid:
    """Return the supported global grid described by an xarray object.

    A regular latitude--longitude field is accepted as CC only when it has
    both poles, equally spaced latitude values, and a complete cyclic
    longitude coordinate.  Other regular grids are rejected rather than
    being reinterpreted as CC.
    """
    coordinates = find_horizontal_coordinates(field)
    latitude = _coordinate_values(field, coordinates.latitude_name)
    longitude = _coordinate_values(field, coordinates.longitude_name)
    _validate_longitude(longitude)

    if _matches_gl_latitudes(latitude):
        return Grid("gl", latitude, longitude)
    if _matches_cc_latitudes(latitude):
        return Grid("cc", latitude, longitude)
    raise ValueError(
        "unsupported latitude coordinate: expected DUCC Gauss--Legendre "
        "nodes or a pole-including equally spaced Clenshaw--Curtis grid"
    )


def find_horizontal_coordinates(
    field: xr.DataArray | xr.Dataset,
) -> HorizontalCoordinates:
    """Locate unambiguous one-dimensional latitude and longitude coordinates.

    Exact CF ``standard_name`` metadata takes precedence over the canonical
    coordinate names ``lat``/``latitude`` and ``lon``/``longitude``.  When
    installed, cf-xarray is consulted only after those core paths.
    """
    latitude_name = _find_coordinate(field, "latitude", _LATITUDE_NAMES)
    longitude_name = _find_coordinate(field, "longitude", _LONGITUDE_NAMES)
    latitude = field.coords[latitude_name]
    longitude = field.coords[longitude_name]
    _validate_coordinate(latitude_name, latitude, "latitude")
    _validate_coordinate(longitude_name, longitude, "longitude")
    latitude_dim = str(latitude.dims[0])
    longitude_dim = str(longitude.dims[0])
    if latitude_dim == longitude_dim:
        raise ValueError(
            "latitude and longitude coordinates must use distinct dimensions"
        )
    return HorizontalCoordinates(
        latitude_name=latitude_name,
        longitude_name=longitude_name,
        latitude_dim=latitude_dim,
        longitude_dim=longitude_dim,
    )


def grid_layout(grid: Grid) -> GridLayout:
    """Return the north-to-south and cyclic-eastward order DUCC expects."""
    latitude_indices = np.argsort(-grid.latitude, kind="stable").astype(np.intp)
    latitude_values = grid.latitude[latitude_indices]

    normalized_longitude = np.mod(grid.longitude, 360.0)
    longitude_indices = np.argsort(normalized_longitude, kind="stable").astype(np.intp)
    longitude_values = normalized_longitude[longitude_indices]
    return GridLayout(
        latitude=AxisLayout(
            canonical_indices=latitude_indices,
            restore_indices=np.argsort(latitude_indices).astype(np.intp),
            canonical_values=latitude_values,
        ),
        longitude=AxisLayout(
            canonical_indices=longitude_indices,
            restore_indices=np.argsort(longitude_indices).astype(np.intp),
            canonical_values=longitude_values,
        ),
    )


def grid_capabilities(grid: Grid) -> GridCapabilities:
    """Return the scalar analysis limits for the installed DUCC geometry.

    DUCC documents ``ntheta - 2`` as the CC latitude limit and ``ntheta - 1``
    for GL.  The azimuthal limit is ``(nphi - 1) // 2`` for both geometries.
    """
    latitude_lmax = grid.nlat - 2 if grid.kind == "cc" else grid.nlat - 1
    longitude_mmax = (grid.nlon - 1) // 2
    return GridCapabilities(latitude_lmax, longitude_mmax)


def grids_equivalent(left: Grid, right: Grid) -> bool:
    """Whether two descriptors represent the same sampling points and geometry."""
    if left.kind != right.kind or left.nlat != right.nlat or left.nlon != right.nlon:
        return False
    left_layout = grid_layout(left)
    right_layout = grid_layout(right)
    return bool(
        np.allclose(
            left_layout.latitude.canonical_values,
            right_layout.latitude.canonical_values,
            rtol=0.0,
            atol=_coordinate_tolerance(left.latitude),
        )
        and np.allclose(
            left_layout.longitude.canonical_values,
            right_layout.longitude.canonical_values,
            rtol=0.0,
            atol=_coordinate_tolerance(left.longitude),
        )
    )


def coordinate_attributes(axis: Literal["latitude", "longitude"]) -> dict[str, str]:
    """Return ordinary CF metadata for generated horizontal coordinates."""
    if axis == "latitude":
        return {
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        }
    return {
        "standard_name": "longitude",
        "units": "degrees_east",
        "axis": "X",
    }


def _find_coordinate(
    field: xr.DataArray | xr.Dataset,
    standard_name: Literal["latitude", "longitude"],
    canonical_names: tuple[str, str],
) -> str:
    one_dimensional = [
        (str(name), coordinate)
        for name, coordinate in field.coords.items()
        if coordinate.ndim == 1
    ]
    cf_matches = [
        name
        for name, coordinate in one_dimensional
        if coordinate.attrs.get("standard_name") == standard_name
    ]
    if cf_matches:
        return _unique_coordinate(cf_matches, standard_name, "CF standard_name")

    canonical_matches = [name for name, _ in one_dimensional if name in canonical_names]
    if canonical_matches:
        return _unique_coordinate(canonical_matches, standard_name, "canonical name")

    cf_xarray_matches = _cf_xarray_coordinates(field, standard_name)
    if cf_xarray_matches:
        return _unique_coordinate(cf_xarray_matches, standard_name, "cf-xarray")
    raise ValueError(
        f"could not identify {standard_name} coordinate; add an exact CF "
        f"standard_name or use one of {canonical_names!r}"
    )


def _cf_xarray_coordinates(
    field: xr.DataArray | xr.Dataset, standard_name: str
) -> list[str]:
    """Ask cf-xarray for a coordinate when the optional package is installed."""
    try:
        import cf_xarray  # noqa: F401  # registers the xarray .cf accessor
    except ImportError:
        return []
    try:
        accessor = field.cf  # type: ignore[attr-defined]
    except (AttributeError, ImportError):
        return []
    try:
        values = accessor.coordinates.get(standard_name, ())
    except (AttributeError, KeyError):
        return []
    return [str(value) for value in values if str(value) in field.coords]


def _unique_coordinate(candidates: list[str], axis: str, source: str) -> str:
    unique = sorted(set(candidates))
    if len(unique) != 1:
        joined = ", ".join(unique)
        raise ValueError(f"ambiguous {axis} coordinate from {source}: {joined}")
    return unique[0]


def _validate_coordinate(name: str, coordinate: xr.DataArray, axis: str) -> None:
    if coordinate.ndim != 1 or len(coordinate.dims) != 1:
        raise ValueError(f"{axis} coordinate {name!r} must be one-dimensional")
    values = np.asarray(coordinate.values)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{axis} coordinate {name!r} must be numeric")


def _coordinate_values(
    field: xr.DataArray | xr.Dataset, name: str
) -> NDArray[np.float64]:
    values = np.asarray(field.coords[name].values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"coordinate {name!r} must be one-dimensional")
    return values


def _validate_shape_arguments(nlat: int, nlon: int) -> None:
    if isinstance(nlat, bool) or not isinstance(nlat, int) or nlat < 2:
        raise ValueError("nlat must be an integer of at least 2")
    if isinstance(nlon, bool) or not isinstance(nlon, int) or nlon < 2:
        raise ValueError("nlon must be an integer of at least 2")


def _validate_latitude_order(value: str) -> None:
    if value not in {"ascending", "descending"}:
        raise ValueError("latitude_order must be 'ascending' or 'descending'")


def _longitudes(nlon: int, lon0: float) -> NDArray[np.float64]:
    if not np.isfinite(lon0):
        raise ValueError("lon0 must be finite")
    return float(lon0) + np.arange(nlon, dtype=np.float64) * (360.0 / nlon)


def _validate_grid_arrays(
    kind: GridKind, latitude: NDArray[np.float64], longitude: NDArray[np.float64]
) -> None:
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("grid latitude and longitude must be one-dimensional")
    _validate_shape_arguments(int(latitude.size), int(longitude.size))
    if not np.isfinite(latitude).all():
        raise ValueError("latitude values must be finite")
    if kind == "gl":
        valid_latitude = _matches_gl_latitudes(latitude)
    else:
        valid_latitude = _matches_cc_latitudes(latitude)
    if not valid_latitude:
        label = "Gauss--Legendre" if kind == "gl" else "pole-including CC"
        raise ValueError(f"latitude values do not define a {label} grid")
    _validate_longitude(longitude)


def _matches_cc_latitudes(latitude: NDArray[np.float64]) -> bool:
    if latitude.size < 2 or not np.isfinite(latitude).all():
        return False
    ascending = _ascending_latitudes(latitude)
    expected = np.linspace(-90.0, 90.0, latitude.size, dtype=np.float64)
    return bool(
        np.allclose(
            ascending,
            expected,
            rtol=0.0,
            atol=_coordinate_tolerance(latitude),
        )
    )


def _matches_gl_latitudes(latitude: NDArray[np.float64]) -> bool:
    if latitude.size < 2 or not np.isfinite(latitude).all():
        return False
    try:
        ascending = _ascending_latitudes(latitude)
    except ValueError:
        return False
    import ducc0

    expected_descending = 90.0 - np.rad2deg(ducc0.misc.GL_thetas(latitude.size))
    expected = expected_descending[::-1]
    return bool(
        np.allclose(
            ascending,
            expected,
            rtol=0.0,
            atol=_coordinate_tolerance(latitude),
        )
    )


def _ascending_latitudes(latitude: NDArray[np.float64]) -> NDArray[np.float64]:
    difference = np.diff(latitude)
    if np.all(difference > 0.0):
        return latitude
    if np.all(difference < 0.0):
        return latitude[::-1]
    raise ValueError("latitude values must be strictly ascending or descending")


def _validate_longitude(longitude: NDArray[np.float64]) -> None:
    if longitude.ndim != 1 or longitude.size < 2:
        raise ValueError("longitude must contain at least two values")
    if not np.isfinite(longitude).all():
        raise ValueError("longitude values must be finite")
    normalized = np.mod(longitude, 360.0)
    sorted_values = np.sort(normalized)
    tolerance = _coordinate_tolerance(longitude)
    if np.any(np.diff(sorted_values) <= tolerance):
        raise ValueError(
            "longitude has a duplicated cyclic endpoint or duplicate value"
        )
    spacing = 360.0 / longitude.size
    cyclic_steps = np.diff(np.concatenate((sorted_values, sorted_values[:1] + 360.0)))
    if not np.allclose(cyclic_steps, spacing, rtol=0.0, atol=tolerance):
        raise ValueError(
            "longitude must be uniformly spaced over one complete non-duplicated cycle"
        )


def _coordinate_tolerance(values: NDArray[np.float64]) -> float:
    """Coordinate tolerance that permits ordinary float32 coordinate storage."""
    finite = np.asarray(values, dtype=np.float64)
    scale = max(360.0, float(np.max(np.abs(finite), initial=0.0)))
    return float(max(1.0e-8, 8.0 * np.finfo(np.float32).eps * scale))
