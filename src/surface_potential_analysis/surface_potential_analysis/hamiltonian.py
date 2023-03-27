from typing import Any, Generic, Literal, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis import (
    Basis,
    FundamentalBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
    is_basis_type,
)
from surface_potential_analysis.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfigUtil,
    PositionBasisConfigUtil,
)
from surface_potential_analysis.interpolation import pad_ft_points

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)


_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)


HamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov], np.dtype[np.complex_] | np.dtype[np.float_]
]


class Hamiltonian(TypedDict, Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
    # We need higher kinded types, and const generics to do this properly
    array: HamiltonianPoints[int, int]


MomentumBasisHamiltonian = Hamiltonian[
    MomentumBasis[_L0Cov], MomentumBasis[_L1Cov], MomentumBasis[_L2Cov]
]
PositionBasisHamiltonian = Hamiltonian[
    PositionBasis[_L0Cov], PositionBasis[_L1Cov], PositionBasis[_L2Cov]
]

StackedHamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov, _L2Cov, _L0Cov, _L1Cov, _L2Cov],
    np.dtype[np.complex_] | np.dtype[np.float_],
]


class StackedHamiltonian(TypedDict, Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
    # We need higher kinded types to do this properly
    array: StackedHamiltonianPoints[int, int, int]


MomentumBasisStackedHamiltonian = StackedHamiltonian[
    MomentumBasis[_L0Cov], MomentumBasis[_L1Cov], MomentumBasis[_L2Cov]
]
PositionBasisStackedHamiltonian = StackedHamiltonian[
    PositionBasis[_L0Cov], PositionBasis[_L1Cov], PositionBasis[_L2Cov]
]

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_CBX0Inv = TypeVar("_CBX0Inv", bound=FundamentalBasis[Any])
_CBX1Inv = TypeVar("_CBX1Inv", bound=FundamentalBasis[Any])
_CBX2Inv = TypeVar("_CBX2Inv", bound=FundamentalBasis[Any])


def flatten_hamiltonian(
    hamiltonian: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]
) -> Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]:
    n_states = np.prod(hamiltonian["array"].shape[:3])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(n_states, n_states),
    }


def stack_hamiltonian(
    hamiltonian: Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]
) -> StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]:
    basis = BasisConfigUtil(hamiltonian["basis"])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(*basis.shape, *basis.shape),
    }


def convert_stacked_hamiltonian_to_momentum_basis(
    hamiltonian: PositionBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    transformed = np.fft.ifftn(
        np.fft.fftn(hamiltonian["array"], axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )
    util = PositionBasisConfigUtil(hamiltonian["basis"])
    return {
        "basis": util.get_reciprocal_basis(),
        "array": transformed,
    }


def convert_hamiltonian_to_momentum_basis(
    hamiltonian: PositionBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    stacked = stack_hamiltonian(hamiltonian)
    converted = convert_stacked_hamiltonian_to_momentum_basis(stacked)
    return flatten_hamiltonian(converted)


def convert_stacked_hamiltonian_to_position_basis(
    hamiltonian: MomentumBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisStackedHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    # TODO: which way round
    transformed = np.fft.fftn(
        np.fft.ifftn(hamiltonian["array"], axes=(0, 1, 2), norm="ortho"),
        axes=(3, 4, 5),
        norm="ortho",
    )
    util = MomentumBasisConfigUtil(hamiltonian["basis"])
    return {
        "basis": util.get_reciprocal_basis(),
        "array": transformed,
    }


def convert_hamiltonian_to_position_basis(
    hamiltonian: MomentumBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisHamiltonian[_L0Inv, _L1Inv, _L2Inv]:
    stacked = stack_hamiltonian(hamiltonian)
    converted = convert_stacked_hamiltonian_to_position_basis(stacked)
    return flatten_hamiltonian(converted)


@overload
def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    len: _L0Inv,
    axis: Literal[0, -3],
) -> StackedHamiltonian[TruncatedBasis[_L0Inv, _CBX0Inv], _CBX1Inv, _CBX2Inv]:
    ...


@overload
def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    len: _L0Inv,
    axis: Literal[1, -2],
) -> StackedHamiltonian[_CBX0Inv, TruncatedBasis[_L0Inv, _CBX1Inv], _CBX2Inv]:
    ...


@overload
def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    len: _L0Inv,
    axis: Literal[2, -1],
) -> StackedHamiltonian[_CBX0Inv, _CBX1Inv, TruncatedBasis[_L0Inv, _CBX2Inv]]:
    ...


def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    len: _L0Inv,
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> (
    StackedHamiltonian[TruncatedBasis[_L0Inv, _CBX0Inv], _CBX1Inv, _CBX2Inv]
    | StackedHamiltonian[_CBX0Inv, TruncatedBasis[_L0Inv, _CBX1Inv], _CBX2Inv]
    | StackedHamiltonian[_CBX0Inv, _CBX1Inv, TruncatedBasis[_L0Inv, _CBX2Inv]]
):
    parent_basis: FundamentalBasis[Any] = hamiltonian["basis"][axis % 3]
    if not is_basis_type(parent_basis, "momentum"):
        raise NotImplementedError()
    padded = pad_ft_points(
        hamiltonian["array"], s=(len, len), axes=(axis % 3, 3 + (axis % 3))
    )
    new_basis: TruncatedBasis[Any, Any] = {
        "n": len,
        "_type": "truncated",
        "parent": parent_basis,
    }
    basis: list[Basis[Any, Any]] = list(hamiltonian["basis"])
    basis[axis] = new_basis
    return {"array": padded, "basis": tuple(basis)}  # type: ignore


@overload
def expand_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[TruncatedBasis[int, _CBX0Inv], _BX1Inv, _BX2Inv],
    axis: Literal[0, -3],
) -> StackedHamiltonian[_CBX0Inv, _BX1Inv, _BX2Inv]:
    ...


@overload
def expand_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[_BX0Inv, TruncatedBasis[int, _CBX1Inv], _BX2Inv],
    axis: Literal[1, -2],
) -> StackedHamiltonian[_BX0Inv, _CBX1Inv, _BX2Inv]:
    ...


@overload
def expand_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[_BX0Inv, _BX1Inv, TruncatedBasis[int, _CBX2Inv]],
    axis: Literal[2, -1],
) -> StackedHamiltonian[_BX0Inv, _BX1Inv, _CBX2Inv]:
    ...


def expand_hamiltonian_basis(
    hamiltonian: StackedHamiltonian[TruncatedBasis[int, _CBX0Inv], _BX1Inv, _BX2Inv]
    | StackedHamiltonian[_BX0Inv, TruncatedBasis[int, _CBX1Inv], _BX2Inv]
    | StackedHamiltonian[_BX0Inv, _BX1Inv, TruncatedBasis[int, _CBX2Inv]],
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> StackedHamiltonian[_BX0Inv | _CBX0Inv, _BX1Inv | _CBX1Inv, _BX2Inv | _CBX2Inv]:
    raise NotImplementedError()


_BX = TypeVar("_BX", bound=Basis[Any, Any], contravariant=True)


@overload
def hamiltonian_axis_in_basis(
    hamiltonian: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
    basis: _BX,
    axis: Literal[0, -3],
) -> StackedHamiltonian[_BX, _BX1Inv, _BX2Inv]:
    ...


@overload
def hamiltonian_axis_in_basis(
    hamiltonian: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
    basis: _BX,
    axis: Literal[1, -2],
) -> StackedHamiltonian[_BX0Inv, _BX, _BX2Inv]:
    ...


@overload
def hamiltonian_axis_in_basis(
    hamiltonian: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
    basis: _BX,
    axis: Literal[2, -1],
) -> StackedHamiltonian[_BX0Inv, _BX1Inv, _BX]:
    ...


def hamiltonian_axis_in_basis(
    hamiltonian: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
    basis: _BX,
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> (
    StackedHamiltonian[_BX, _BX1Inv, _BX2Inv]
    | StackedHamiltonian[_BX0Inv, _BX, _BX2Inv]
    | StackedHamiltonian[_BX0Inv, _BX1Inv, _BX]
):
    raise NotImplementedError()


def stacked_hamiltonian_in_basis(
    hamiltonian: StackedHamiltonian[Any, Any, Any],
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
) -> StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]:
    """
    Transform a stacked hamiltonian into the given basis

    Parameters
    ----------
    hamiltonian : StackedHamiltonian[BX0, BX1, BX2]
    basis : BasisConfig[BX0, BX1, BX2]

    Returns
    -------
    StackedHamiltonian[BX0, BX1, BX2]
    """
    x0_transformed = hamiltonian_axis_in_basis(hamiltonian, basis[0], 0)
    x1_transformed = hamiltonian_axis_in_basis(x0_transformed, basis[1], 1)
    return hamiltonian_axis_in_basis(x1_transformed, basis[2], 2)


def hamiltonian_in_basis(
    hamiltonian: Hamiltonian[Any, Any, Any],
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
) -> Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]:
    """
    Transform a hamiltonian into the given basis

    Parameters
    ----------
    hamiltonian : Hamiltonian
    basis : BasisConfig[BX0, BX1, BX2]

    Returns
    -------
    Hamiltonian[BX0, BX1, BX2]
    """
    stacked = stack_hamiltonian(hamiltonian)
    converted = stacked_hamiltonian_in_basis(stacked, basis)
    return flatten_hamiltonian(converted)


H = TypeVar("H", bound=Hamiltonian[Any, Any, Any])


def add_hamiltonian_stacked(
    a: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
    b: StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv],
) -> StackedHamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]:
    return {"basis": a["basis"], "array": a["array"] + b["array"]}


def add_hamiltonian(
    a: Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv], b: Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]
) -> Hamiltonian[_BX0Inv, _BX1Inv, _BX2Inv]:
    return {"basis": a["basis"], "array": a["array"] + b["array"]}
