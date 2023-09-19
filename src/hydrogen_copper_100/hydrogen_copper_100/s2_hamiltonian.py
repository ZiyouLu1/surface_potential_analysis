from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.hamiltonian_builder import (
    explicit_z_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.stacked_basis.potential_basis import (
    PotentialBasisConfig,
    select_minimum_potential_3d,
)
from surface_potential_analysis.util.constants import HYDROGEN_MASS
from surface_potential_analysis.util.decorators import timed

from .s1_potential import (
    get_interpolated_potential,
    get_interpolated_potential_relaxed,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        ExplicitBasis,
        ExplicitBasis3d,
        FundamentalPositionBasis1d,
        TransformedPositionBasis,
        TransformedPositionBasis3d,
    )
    from surface_potential_analysis.axis.stacked_axis import StackedBasisLike
    from surface_potential_analysis.operator import SingleBasisOperator
    from surface_potential_analysis.stacked_basis.sho_basis import SHOBasisConfig

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
    StackedBasisLike[
        TransformedPositionBasis3d[_L0, _L3],
        TransformedPositionBasis3d[_L1, _L4],
        ExplicitBasis3d[_L2, _L5],
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
    potential["data"] = 0.5 * (
        potential["data"] + potential["data"].reshape(shape).swapaxes(0, 1).reshape(-1)
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
    StackedBasisLike[
        TransformedPositionBasis[_L3, _L3, Literal[3]],
        TransformedPositionBasis[_L4, _L4, Literal[3]],
        ExplicitBasis[_L2, _L5, Literal[3]],
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
    vector = potential["data"].reshape(*shape)
    np.testing.assert_equal(vector, vector[::-1])
    np.testing.assert_equal(vector, vector[:, ::-1])
    np.testing.assert_equal(vector, vector[::-1, ::-1])
    np.testing.assert_equal(vector, vector.swapaxes(0, 1))
    np.testing.assert_equal(vector, vector.swapaxes(0, 1)[:, ::-1])
    min_pot = np.unravel_index(np.argmin(potential["data"]), shape)
    potential["data"] = np.roll(
        vector, (-min_pot[0], -min_pot[1], -min_pot[2]), (0, 1, 2)
    ).reshape(-1)
    config: PotentialBasisConfig[FundamentalPositionBasis1d[_L2], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),  # type: ignore[arg-type]
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )


def get_hamiltonian_relaxed(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> SingleBasisOperator[
    StackedBasisLike[
        TransformedPositionBasis[_L3, _L3, Literal[3]],
        TransformedPositionBasis[_L4, _L4, Literal[3]],
        ExplicitBasis[_L2, _L5, Literal[3]],
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
    config: PotentialBasisConfig[FundamentalPositionBasis1d[_L2], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),  # type: ignore[arg-type]
    }
    return explicit_z_basis.total_surface_hamiltonian_as_fundamental(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )
