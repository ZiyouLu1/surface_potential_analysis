import math
from typing import Any, Literal, TypedDict, TypeVar

import numpy as np
import scipy
import scipy.special
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import BasisWithLength
from surface_potential_analysis.basis_config import PositionBasisConfig
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.hamiltonian_eigenstates import calculate_eigenstates
from surface_potential_analysis.potential.potential import Potential

from .basis import (
    BasisUtil,
    ExplicitBasis,
    FundamentalBasis,
    MomentumBasis,
    PositionBasis,
)

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


def calculate_sho_wavefunction(
    x_points: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    sho_omega: float,
    mass: float,
    n: int,
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_x = x_points * norm

    prefactor = math.sqrt((norm / (2**n)) / (math.factorial(n) * math.sqrt(math.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_x)
    exponential = np.exp(-np.square(normalized_x) / 2)
    return prefactor * hermite * exponential  # type: ignore


def calculate_x_distances(
    parent: BasisWithLength[_L0Inv, Any],
    x_origin: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    util = BasisUtil(parent)
    x_points = util.fundamental_x_points

    x0_norm = util.delta_x.copy() / np.linalg.norm(util.delta_x)
    distances_origin = np.dot(x0_norm, x_origin)
    x_distances = np.dot(x0_norm, x_points) + distances_origin
    return x_distances  # type: ignore


class SHOBasisConfig(TypedDict):
    sho_omega: float
    mass: float
    x_origin: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]


def sho_basis_from_config(
    parent: FundamentalBasis[_L1Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis[_L0Inv, MomentumBasis[_L1Inv]]:
    """
    Calculate the exact sho basis for a given basis, by directly
    diagonalizing the sho wavefunction in this basis. The resulting wavefunction
    is guaranteed to be orthonormal in this basis

    Parameters
    ----------
    parent : _FBInv
        _description_
    config : SHOBasisConfig
        _description_
    n : _L0Inv
        _description_

    Returns
    -------
    ExplicitBasis[_L0Inv, _FBInv]
        _description_
    """

    util = BasisUtil(parent)

    delta_x1 = (
        np.array([0, 1, 0])
        if np.allclose([1, 0, 0], parent["delta_x"])
        else np.array([1, 0, 0])
    )
    delta_x2 = np.cross(parent["delta_x"], delta_x1)
    delta_x2 /= np.linalg.norm(delta_x2)

    potential_basis: PositionBasisConfig[_L1Inv, Literal[1], Literal[1]] = (
        util.get_fundamental_basis_in("position"),
        {"_type": "position", "delta_x": delta_x1, "n": 1},
        {"_type": "position", "delta_x": delta_x2, "n": 1},
    )
    x_distances = calculate_x_distances(parent, config["x_origin"])

    sho_potential: Potential[_L1Inv, Literal[1], Literal[1]] = {
        "basis": potential_basis,
        "points": 0.5 * config["mass"] * config["sho_omega"] ** 2 * x_distances**2,
    }
    hamiltonian = total_surface_hamiltonian(
        sho_potential, config["mass"], np.array([0, 0, 0])
    )
    eigenstates = calculate_eigenstates(hamiltonian)

    vectors = eigenstates["vectors"][:n]
    return {"_type": "explicit", "parent": eigenstates["basis"][0], "vectors": vectors}


def infinate_sho_basis_from_config(
    parent: PositionBasis[_L1Inv], config: SHOBasisConfig, n: _L0Inv
) -> ExplicitBasis[_L0Inv, PositionBasis[_L1Inv]]:
    x_distances = calculate_x_distances(parent, config["x_origin"])

    vectors = np.array(
        [
            calculate_sho_wavefunction(
                x_distances, n=i, mass=config["mass"], sho_omega=config["sho_omega"]
            )
            for i in range(n)
        ]
    )
    util = BasisUtil(parent)
    normalized = vectors * np.linalg.norm(util.fundamental_dx)
    return {"_type": "explicit", "parent": parent, "vectors": normalized}
