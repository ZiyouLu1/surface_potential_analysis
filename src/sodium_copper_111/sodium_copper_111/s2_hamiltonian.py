from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)

from .s1_potential import get_interpolated_potential, get_interpolated_potential_2d

if TYPE_CHECKING:
    import numpy as np
    from surface_potential_analysis.basis.basis import (
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)

SODIUM_MASS = 3.8175458e-26
HYDROGEN_MASS = 1.6735575e-27
LITHIUM_MASS = 1.1525801e-26


def get_hamiltonian(
    shape: tuple[_L0Inv],
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]],
]:
    potential = get_interpolated_potential(shape)
    return total_surface_hamiltonian(potential, SODIUM_MASS, bloch_fraction)


def get_hamiltonian_hydrogen(
    shape: tuple[_L0Inv],
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]],
]:
    potential = get_interpolated_potential(shape)
    return total_surface_hamiltonian(potential, HYDROGEN_MASS, bloch_fraction)


def get_hamiltonian_2d(
    shape: tuple[_L0Inv, _L1Inv],
    bloch_fraction: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    StackedBasisLike[
        FundamentalTransformedPositionBasis[_L0Inv, Literal[2]],
        FundamentalTransformedPositionBasis[_L1Inv, Literal[2]],
    ],
]:
    potential = get_interpolated_potential_2d(shape)
    return total_surface_hamiltonian(potential, SODIUM_MASS, bloch_fraction)


def get_hamiltonian_flat(
    shape: tuple[_L0Inv],
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]],
]:
    potential = get_interpolated_potential(shape)
    potential["data"][:] = 0
    return total_surface_hamiltonian(potential, SODIUM_MASS, bloch_fraction)


def get_hamiltonian_flat_lithium(
    shape: tuple[_L0Inv],
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    StackedBasisLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]],
]:
    potential = get_interpolated_potential(shape)
    potential["data"][:] = 0
    return total_surface_hamiltonian(potential, LITHIUM_MASS, bloch_fraction)


def get_hamiltonian_flat_2d(
    shape: tuple[_L0Inv, _L1Inv],
    bloch_fraction: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    StackedBasisLike[
        FundamentalTransformedPositionBasis[_L0Inv, Literal[2]],
        FundamentalTransformedPositionBasis[_L1Inv, Literal[2]],
    ],
]:
    potential = get_interpolated_potential_2d(shape)
    potential["data"][:] = 0
    return total_surface_hamiltonian(potential, SODIUM_MASS, bloch_fraction)
