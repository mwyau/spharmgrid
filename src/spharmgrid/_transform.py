"""Backend-neutral spherical harmonic transform descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Truncation = Literal["triangular", "trapezoidal", "rhomboidal"]


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Inclusive spectral bounds and coefficient-domain geometry.

    ``lmin`` is the lowest retained total degree, while ``lmax`` and ``mmax``
    are the inclusive total-degree and zonal-order transform limits.  A
    rhomboidal mask derives its symmetric limit as ``lmax - mmax``.
    """

    lmin: int
    lmax: int
    mmax: int
    truncation: Truncation = "triangular"

    def __post_init__(self) -> None:
        if (
            isinstance(self.lmin, bool)
            or isinstance(self.lmax, bool)
            or isinstance(self.mmax, bool)
            or not isinstance(self.lmin, int)
            or not isinstance(self.lmax, int)
            or not isinstance(self.mmax, int)
        ):
            raise TypeError("transform limits must be integers")
        if self.lmin < 0 or self.lmin > self.lmax:
            raise ValueError("transform limits must satisfy 0 <= lmin <= lmax")
        if self.mmax < 0 or self.mmax > self.lmax:
            raise ValueError("transform limits must satisfy 0 <= mmax <= lmax")
        if not isinstance(self.truncation, str) or self.truncation not in {
            "triangular",
            "trapezoidal",
            "rhomboidal",
        }:
            raise ValueError(
                "truncation must be 'triangular', 'trapezoidal', or 'rhomboidal'"
            )
        if self.truncation == "triangular" and self.mmax != self.lmax:
            raise ValueError("triangular transforms require mmax=lmax")
