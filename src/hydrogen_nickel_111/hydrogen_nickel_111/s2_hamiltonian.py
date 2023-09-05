from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from surface_potential_analysis.basis.potential_basis import select_minimum_potential_3d
from surface_potential_analysis.hamiltonian_builder import (
    explicit_z_basis,
)
from surface_potential_analysis.potential.potential import truncate_potential
from surface_potential_analysis.util.constants import DEUTERIUM_MASS, HYDROGEN_MASS

from .s1_potential import (
    get_interpolated_extrapolated_potential,
    get_interpolated_potential,
)

if TYPE_CHECKING:
    import numpy as np
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis,
        FundamentalPositionAxis1d,
        TransformedPositionAxis,
    )
    from surface_potential_analysis.basis.potential_basis import PotentialBasisConfig
    from surface_potential_analysis.operator import SingleBasisOperator

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


def get_hamiltonian_hydrogen(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    tuple[
        TransformedPositionAxis[_L3, _L3, Literal[3]],
        TransformedPositionAxis[_L4, _L4, Literal[3]],
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
    potential["vector"] = 0.5 * (
        potential["vector"] + potential["vector"].reshape(shape).swapaxes(0, 1).ravel()
    )
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L2]], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )


def get_hamiltonian_hydrogen_extrapolated(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    tuple[
        TransformedPositionAxis[_L3, _L3, Literal[3]],
        TransformedPositionAxis[_L4, _L4, Literal[3]],
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
    potential = get_interpolated_extrapolated_potential(shape)
    potential = truncate_potential(potential, cutoff=3.5e-19, n=5, offset=1e-20)
    potential["vector"] = 0.5 * (
        potential["vector"] + potential["vector"].reshape(shape).swapaxes(0, 1).ravel()
    )
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L2]], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )


def get_hamiltonian_deuterium(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    tuple[
        TransformedPositionAxis[_L3, _L3, Literal[3]],
        TransformedPositionAxis[_L4, _L4, Literal[3]],
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
    potential["vector"] = 0.5 * (
        potential["vector"] + potential["vector"].reshape(shape).swapaxes(0, 1).ravel()
    )
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L2]], _L5] = {
        "n": resolution[2],
        "mass": DEUTERIUM_MASS,
        "potential": select_minimum_potential_3d(potential),
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )
