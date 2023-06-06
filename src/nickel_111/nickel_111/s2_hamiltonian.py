from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.sho_basis import (
    infinate_sho_basis_from_config,
    sho_basis_from_config,
)
from surface_potential_analysis.hamiltonian_builder import sho_subtracted_basis
from surface_potential_analysis.util.decorators import timed

from .s1_potential import get_interpolated_potential

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        ExplicitAxis3d,
        MomentumAxis3d,
    )
    from surface_potential_analysis.basis.sho_basis import SHOBasisConfig
    from surface_potential_analysis.hamiltonian import HamiltonianWith3dBasis

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
    MomentumAxis3d[_L0, _L3],
    MomentumAxis3d[_L1, _L4],
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
        "mass": 1.6735575e-27,
        "x_origin": np.array([0, 0, -1.0000000000000004e-10]),
    }
    return sho_subtracted_basis.total_surface_hamiltonian(
        potential, config, bloch_fraction, resolution
    )


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
        infinate_sho_basis_from_config(parent, config, n),
        sho_basis_from_config(parent, config, n),
    )
