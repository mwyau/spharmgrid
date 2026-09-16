"""Backend-neutral spherical-harmonic transform descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """A scalar/vector transform bandwidth supported by its source geometry."""

    lmax: int
    mmax: int

    def __post_init__(self) -> None:
        if self.lmax < 0 or self.mmax < 0 or self.mmax > self.lmax:
            raise ValueError("transform limits must satisfy 0 <= mmax <= lmax")
