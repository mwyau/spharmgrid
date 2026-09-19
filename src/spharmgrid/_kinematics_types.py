# SPDX-FileCopyrightText: Albert M. W. Yau
#
# SPDX-License-Identifier: BSD-3-Clause

"""Type aliases for kinematic source representations."""

from __future__ import annotations

from typing import Literal, TypeAlias

ScalarSource: TypeAlias = Literal[
    "vorticity", "divergence", "streamfunction", "velocity_potential"
]
RotationalWindSource: TypeAlias = Literal["vorticity", "streamfunction"]
DivergentWindSource: TypeAlias = Literal["divergence", "velocity_potential"]
WindSource: TypeAlias = Literal["vorticity_divergence", "potentials"]

__all__ = [
    "DivergentWindSource",
    "RotationalWindSource",
    "ScalarSource",
    "WindSource",
]
