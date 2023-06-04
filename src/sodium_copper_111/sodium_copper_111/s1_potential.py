from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from scipy.constants import electron_volt
from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis1d,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import Potential1d

_L0Inv = TypeVar("_L0Inv", bound=int)

LATTICE_CONSTANT = 3.615 * 10**-10
BARRIER_ENERGY = 55 * 10**-3 * electron_volt


def get_potential() -> Potential1d[tuple[FundamentalMomentumAxis1d[Literal[3]]]]:
    delta_x = np.sqrt(3) * LATTICE_CONSTANT / 2
    axis = FundamentalMomentumAxis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * BARRIER_ENERGY * np.array([2, -1, -1]) / np.sqrt(3)
    return {"basis": (axis,), "vector": vector}


def get_interpolated_potential(
    shape: tuple[_L0Inv],
) -> Potential1d[tuple[FundamentalMomentumAxis1d[_L0Inv]]]:
    potential = get_potential()
    axis = FundamentalMomentumAxis1d[_L0Inv](potential["basis"][0].delta_x, shape[0])
    interpolated = pad_ft_points(potential["vector"], s=shape, axes=(0,))
    vector = interpolated * np.sqrt(3) / np.sqrt(shape[0])
    return {"basis": (axis,), "vector": vector}  # type: ignore[typeddict-item]
