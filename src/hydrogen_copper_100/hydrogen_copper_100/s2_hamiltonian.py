from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.potential_basis import (
    PotentialBasisConfig,
    select_minimum_potential_3d,
)
from surface_potential_analysis.hamiltonian_builder import (
    explicit_z_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.util.constants import HYDROGEN_MASS
from surface_potential_analysis.util.decorators import timed

from .s1_potential import (
    get_interpolated_potential,
    get_interpolated_potential_relaxed,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis,
        ExplicitAxis3d,
        FundamentalPositionAxis1d,
        MomentumAxis,
        MomentumAxis3d,
    )
    from surface_potential_analysis.basis.sho_basis import SHOBasisConfig
    from surface_potential_analysis.operator import SingleBasisOperator

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


@timed
def get_hamiltonian_sho(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    tuple[MomentumAxis3d[_L0, _L3], MomentumAxis3d[_L1, _L4], ExplicitAxis3d[_L2, _L5]]
]:
    """
    Generate a Hamiltonian using an infinate SHO basis.

    Parameters
    ----------
    shape : tuple[_L0, _L1, _L2]
        Shape of the initial potential
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        Bloch phase
    resolution : tuple[_L3, _L4, _L5]
        Resolution of the truncated basis in x,y

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
        Hamiltonian in the specified basis
    """
    potential = get_interpolated_potential(shape)
    potential["vector"] = 0.5 * (
        potential["vector"]
        + potential["vector"].reshape(shape).swapaxes(0, 1).reshape(-1)
    )
    config: SHOBasisConfig = {
        "sho_omega": 117905964225836.06,
        "mass": 1.6735575e-27,
        "x_origin": np.array([0, 0, -1.840551985155284e-10]),
    }
    return sho_subtracted_basis.total_surface_hamiltonian(
        potential, config, bloch_fraction, resolution
    )


def get_hamiltonian(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    tuple[
        MomentumAxis[_L3, _L3, Literal[3]],
        MomentumAxis[_L4, _L4, Literal[3]],
        ExplicitAxis[_L2, _L5, Literal[3]],
    ]
]:
    """
    Generate a Hamiltonian using an infinate SHO basis.

    Parameters
    ----------
    shape : tuple[_L0, _L1, _L2]
        Shape of the initial potential
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        Bloch phase
    resolution : tuple[_L3, _L4, _L5]
        Resolution of the truncated basis in x,y

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
        Hamiltonian in the specified basis
    """
    potential = get_interpolated_potential(shape)
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L2]], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )


def get_hamiltonian_relaxed(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    tuple[
        MomentumAxis[_L3, _L3, Literal[3]],
        MomentumAxis[_L4, _L4, Literal[3]],
        ExplicitAxis[_L2, _L5, Literal[3]],
    ]
]:
    """
    Generate a Hamiltonian using an infinate SHO basis.

    Parameters
    ----------
    shape : tuple[_L0, _L1, _L2]
        Shape of the initial potential
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
        Bloch phase
    resolution : tuple[_L3, _L4, _L5]
        Resolution of the truncated basis in x,y

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
        Hamiltonian in the specified basis
    """
    potential = get_interpolated_potential_relaxed(shape)
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L2]], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )
