from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    ExplicitAxis1d,
    FundamentalPositionAxis1d,
    FundamentalPositionAxis3d,
)
from surface_potential_analysis.basis.basis import Basis1d
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenstates_hermitian,
)

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import (
        Potential,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVectorList

_B1d0Cov = TypeVar("_B1d0Cov", covariant=True, bound=Basis1d[Any])
_B1d0Inv = TypeVar("_B1d0Inv", bound=Basis1d[Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Cov = TypeVar("_L1Cov", covariant=True, bound=int)


class PotentialBasisConfig(TypedDict, Generic[_B1d0Cov, _L1Cov]):
    """Configures the generation of an explicit basis from a given potential."""

    potential: Potential[_B1d0Cov]
    mass: float
    n: _L1Cov


_N0Inv = TypeVar("_N0Inv", bound=int)


def get_potential_basis_config_eigenstates(
    config: PotentialBasisConfig[_B1d0Inv, _N0Inv],
) -> StateVectorList[_B1d0Inv]:
    """
    Get the eigenstates of the potential, as used in the final basis.

    Parameters
    ----------
    config : PotentialBasisConfig[_L0Inv]

    Returns
    -------
    EigenstateList[PositionBasis3d[_L0Inv, Literal[1], Literal[1]]]
    """
    hamiltonian = total_surface_hamiltonian(
        config["potential"], config["mass"], np.array([0])
    )
    return calculate_eigenstates_hermitian(
        hamiltonian, subset_by_index=(0, config["n"] - 1)
    )


def get_potential_basis_config_basis(
    config: PotentialBasisConfig[_B1d0Inv, _N0Inv],
) -> ExplicitAxis1d[Any, _N0Inv]:
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
    return ExplicitAxis1d(
        eigenstates["basis"][0].delta_x, eigenstates["vectors"]  # type: ignore[arg-type]
    )


def select_minimum_potential_3d(
    potential: Potential[tuple[Any, Any, FundamentalPositionAxis3d[_L0Inv]]]
) -> Potential[tuple[FundamentalPositionAxis1d[_L0Inv]]]:
    """
    Given a 3D potential in the standard configuration select the minimum potential.

    Parameters
    ----------
    potential : Potential[tuple[Any, Any, FundamentalPositionAxis3d[_L0]]]

    Returns
    -------
    Potential[tuple[FundamentalPositionAxis1d[_L0]]]
    """
    shape = BasisUtil(potential["basis"]).shape
    arg_min = np.unravel_index(np.argmin(potential["vector"]), shape)
    return {
        "basis": (
            FundamentalPositionAxis1d(
                np.array([potential["basis"][2].delta_x[2]]), shape[2]  # type: ignore[arg-type]
            ),
        ),
        "vector": potential["vector"].reshape(shape)[arg_min[0], arg_min[1], :],
    }
