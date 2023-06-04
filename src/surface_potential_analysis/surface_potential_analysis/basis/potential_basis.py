from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    ExplicitAxis3d,
)
from surface_potential_analysis.eigenstate.eigenstate_calculation import (
    calculate_eigenstates,
)
from surface_potential_analysis.hamiltonian.conversion import (
    convert_hamiltonian_to_position_basis,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis3d,
    )
    from surface_potential_analysis.eigenstate.eigenstate import EigenstateList3d
    from surface_potential_analysis.potential.potential import (
        FundamentalPositionBasisPotential3d,
    )

_L0Cov = TypeVar("_L0Cov", covariant=True, bound=int)
_L1Cov = TypeVar("_L1Cov", covariant=True, bound=int)


class PotentialBasisConfig(TypedDict, Generic[_L0Cov, _L1Cov]):
    """Configures the generation of an explicit basis from a given potential."""

    potential: FundamentalPositionBasisPotential3d[_L0Cov, Literal[1], Literal[1]]
    mass: float
    n: _L1Cov


_N0Inv = TypeVar("_N0Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)


def get_potential_basis_config_eigenstates(
    config: PotentialBasisConfig[_NF0Inv, _N0Inv],
) -> EigenstateList3d[FundamentalPositionBasis3d[_NF0Inv, Literal[1], Literal[1]]]:
    """
    Get the eigenstates of the potential, as used in the final basis.

    Parameters
    ----------
    config : PotentialBasisConfig[_L0Inv]

    Returns
    -------
    EigenstateList[PositionBasis3d[_L0Inv, Literal[1], Literal[1]]]
    """
    h = total_surface_hamiltonian(
        config["potential"], config["mass"], np.array([0, 0, 0])
    )
    position_h = convert_hamiltonian_to_position_basis(h)
    return calculate_eigenstates(position_h, subset_by_index=(0, config["n"] - 1))


def get_potential_basis_config_basis(
    config: PotentialBasisConfig[_NF0Inv, _N0Inv],
) -> ExplicitAxis3d[_NF0Inv, _NF0Inv]:
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
    return ExplicitAxis3d(
        eigenstates["basis"][0].delta_x, eigenstates["vectors"]  # type: ignore[arg-type]
    )
