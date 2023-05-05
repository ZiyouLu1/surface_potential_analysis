from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.eigenstate.eigenstate_calculation import (
    calculate_eigenstates,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        PositionBasis,
    )
    from surface_potential_analysis.basis_config.basis_config import PositionBasisConfig
    from surface_potential_analysis.eigenstate.eigenstate import EigenstateList
    from surface_potential_analysis.potential.potential import Potential

_L0Cov = TypeVar("_L0Cov", covariant=True, bound=int)
_L1Cov = TypeVar("_L1Cov", covariant=True, bound=int)


class PotentialBasisConfig(TypedDict, Generic[_L0Cov, _L1Cov]):
    """Configures the generation of an explicit basis from a given potential."""

    potential: Potential[_L0Cov, Literal[1], Literal[1]]
    mass: float
    n: _L1Cov


_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


def get_potential_basis_config_eigenstates(
    config: PotentialBasisConfig[_L0Inv, _L1Inv],
) -> EigenstateList[PositionBasisConfig[_L0Inv, Literal[1], Literal[1]]]:
    """
    Get the eigenstates of the potential, as used in the final basis.

    Parameters
    ----------
    config : PotentialBasisConfig[_L0Inv]

    Returns
    -------
    EigenstateList[PositionBasisConfig[_L0Inv, Literal[1], Literal[1]]]
    """
    hamiltonian = total_surface_hamiltonian(
        config["potential"], config["mass"], np.array([0, 0, 0])
    )
    # TODO:
    return calculate_eigenstates(hamiltonian, subset_by_index=(0, config["n"] - 1))


def get_potential_basis_config_basis(
    config: PotentialBasisConfig[_L0Inv, _L1Inv],
) -> ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]]:
    """
    Get the explicit basis for the potential basis config.

    Parameters
    ----------
    config : PotentialBasisConfig[_L0Inv, _L1Inv]

    Returns
    -------
    ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]]
    """
    eigenstates = get_potential_basis_config_eigenstates(config)
    return {
        "_type": "explicit",
        "parent": eigenstates["basis"][0],
        "vectors": eigenstates["vectors"],  # type: ignore[typeddict-item]
    }
