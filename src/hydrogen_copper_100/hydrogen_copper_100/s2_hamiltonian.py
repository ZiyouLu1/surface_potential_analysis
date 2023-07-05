from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.axis.axis import (
    FundamentalMomentumAxis3d,
    FundamentalPositionAxis1d,
)
from surface_potential_analysis.basis.potential_basis import (
    PotentialBasisConfig,
    select_minimum_potential_3d,
)
from surface_potential_analysis.hamiltonian_builder import (
    explicit_z_basis,
    sho_subtracted_basis,
)
from surface_potential_analysis.util.decorators import timed

from hydrogen_copper_100.constants import HYDROGEN_MASS
from hydrogen_copper_100.s1_potential import (
    get_interpolated_potential,
    get_interpolated_potential_relaxed,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis3d,
        MomentumAxis3d,
    )
    from surface_potential_analysis.basis.sho_basis import SHOBasisConfig
    from surface_potential_analysis.operator import HamiltonianWith3dBasis

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


@timed
def generate_hamiltonian_sho(
    shape: tuple[_L0, _L1, _L2],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> HamiltonianWith3dBasis[
    MomentumAxis3d[_L0, _L3], MomentumAxis3d[_L1, _L4], ExplicitAxis3d[_L2, _L5]
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
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),
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


def get_hamiltonian_relaxed(
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
    potential = get_interpolated_potential_relaxed(shape)
    potential["vector"] = 0.5 * (
        potential["vector"] + potential["vector"].reshape(shape).swapaxes(0, 1).ravel()
    )
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_L0]], _L5] = {
        "n": resolution[2],
        "mass": HYDROGEN_MASS,
        "potential": select_minimum_potential_3d(potential),
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
