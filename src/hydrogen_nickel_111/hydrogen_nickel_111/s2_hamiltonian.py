from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis3d,
    FundamentalPositionAxis1d,
    FundamentalPositionAxis3d,
)
from surface_potential_analysis.basis.sho_basis import (
    infinate_sho_axis_3d_from_config,
    sho_axis_3d_from_config,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.hamiltonian_builder import (
    explicit_z_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.util.decorators import timed

from .constants import HYDROGEN_MASS
from .s1_potential import get_interpolated_potential

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import ExplicitAxis3d
    from surface_potential_analysis.basis.potential_basis import PotentialBasisConfig
    from surface_potential_analysis.basis.sho_basis import SHOBasisConfig
    from surface_potential_analysis.operator import HamiltonianWith3dBasis
    from surface_potential_analysis.potential.potential import (
        Potential,
    )

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


@timed
def get_hamiltonian_hydrogen_sho(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> HamiltonianWith3dBasis[
    FundamentalMomentumAxis3d[_L3],
    FundamentalMomentumAxis3d[_L4],
    ExplicitAxis3d[_L2, _L5],
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
    config: SHOBasisConfig = {
        "sho_omega": 195636899474736.66,
        "mass": HYDROGEN_MASS,
        "x_origin": np.array([0, 0, -1.0000000000000004e-10]),
    }
    hamiltonian = sho_subtracted_basis.total_surface_hamiltonian(
        potential, config, bloch_fraction, resolution
    )
    return {
        "basis": (
            FundamentalMomentumAxis3d(
                hamiltonian["basis"][0].delta_x, hamiltonian["basis"][0].n
            ),
            FundamentalMomentumAxis3d(
                hamiltonian["basis"][1].delta_x, hamiltonian["basis"][1].n
            ),
            hamiltonian["basis"][2],
        ),
        "array": hamiltonian["array"],
        "dual_basis": (
            FundamentalMomentumAxis3d(
                hamiltonian["dual_basis"][0].delta_x, hamiltonian["dual_basis"][0].n
            ),
            FundamentalMomentumAxis3d(
                hamiltonian["dual_basis"][1].delta_x, hamiltonian["dual_basis"][1].n
            ),
            hamiltonian["dual_basis"][2],
        ),
    }


def select_minimum_potential(
    potential: Potential[tuple[Any, Any, FundamentalPositionAxis3d[_L0]]]
) -> Potential[tuple[FundamentalPositionAxis1d[_L0]]]:
    shape = BasisUtil(potential["basis"]).shape
    arg_min = np.unravel_index(np.argmin(potential["vector"]), shape)
    return {
        "basis": (
            FundamentalPositionAxis1d(
                np.array([potential["basis"][2].delta_x[2]]), shape[2]
            ),
        ),
        "vector": potential["vector"].reshape(shape)[arg_min[0], arg_min[1], :],
    }


@timed
def get_hamiltonian_deuterium(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> HamiltonianWith3dBasis[
    FundamentalMomentumAxis3d[_L3],
    FundamentalMomentumAxis3d[_L4],
    ExplicitAxis3d[_L2, _L5],
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
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L0]], _L5] = {
        "n": resolution[2],
        "mass": 3.344494393406961e-27,
        "potential": select_minimum_potential(potential),
    }
    hamiltonian = explicit_z_basis.total_surface_hamiltonian(
        potential, bloch_fraction, (resolution[0], resolution[1]), config
    )
    return {
        "basis": (
            FundamentalMomentumAxis3d(
                hamiltonian["basis"][0].delta_x, hamiltonian["basis"][0].n
            ),
            FundamentalMomentumAxis3d(
                hamiltonian["basis"][1].delta_x, hamiltonian["basis"][1].n
            ),
            hamiltonian["basis"][2],
        ),
        "array": hamiltonian["array"],
        "dual_basis": (
            FundamentalMomentumAxis3d(
                hamiltonian["dual_basis"][0].delta_x, hamiltonian["dual_basis"][0].n
            ),
            FundamentalMomentumAxis3d(
                hamiltonian["dual_basis"][1].delta_x, hamiltonian["dual_basis"][1].n
            ),
            hamiltonian["dual_basis"][2],
        ),
    }


def generate_sho_basis(
    n: int,
) -> tuple[ExplicitAxis3d[int, int], ExplicitAxis3d[int, int]]:
    """Get the SHO basis used for the Nickel surface."""
    config: SHOBasisConfig = {
        "sho_omega": 195636899474736.66,
        "mass": 1.6735575e-27,
        "x_origin": np.array([0, 0, -1.0000000000000004e-10]),
    }
    parent = get_interpolated_potential((1, 1, 1000))["basis"][2]
    return (
        infinate_sho_axis_3d_from_config(parent, config, n),
        sho_axis_3d_from_config(parent, config, n),
    )
