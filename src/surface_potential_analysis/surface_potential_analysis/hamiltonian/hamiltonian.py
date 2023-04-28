from typing import Any, Generic, Literal, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis import Basis, FundamentalBasis, TruncatedBasis
from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    is_basis_type,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
    PositionBasisConfig,
)
from surface_potential_analysis.interpolation import pad_ft_points
from surface_potential_analysis.util import slice_along_axis

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)


_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)


HamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov], np.dtype[np.complex_] | np.dtype[np.float_]
]


class Hamiltonian(TypedDict, Generic[_BC0Cov]):
    """Represents a qm hamiltonian in the given basis."""

    basis: _BC0Cov
    # We need higher kinded types, and const generics to do this properly
    array: HamiltonianPoints[int, int]


HamiltonianWithBasis = Hamiltonian[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]

MomentumBasisHamiltonian = Hamiltonian[MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]]
PositionBasisHamiltonian = Hamiltonian[PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]]

_StackedHamiltonianPoints = np.ndarray[
    tuple[_L0Cov, _L1Cov, _L2Cov, _L0Cov, _L1Cov, _L2Cov],
    np.dtype[np.complex_] | np.dtype[np.float_],
]


class StackedHamiltonian(TypedDict, Generic[_BC0Cov]):
    basis: _BC0Cov
    # We need higher kinded types to do this properly
    array: _StackedHamiltonianPoints[int, int, int]


StackedHamiltonianWithBasis = StackedHamiltonian[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]
MomentumBasisStackedHamiltonian = StackedHamiltonian[
    MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]
]
PositionBasisStackedHamiltonian = StackedHamiltonian[
    PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
]

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LInv = TypeVar("_LInv", bound=int)

_CBX0Inv = TypeVar("_CBX0Inv", bound=FundamentalBasis[Any])
_CBX1Inv = TypeVar("_CBX1Inv", bound=FundamentalBasis[Any])
_CBX2Inv = TypeVar("_CBX2Inv", bound=FundamentalBasis[Any])


def flatten_hamiltonian(
    hamiltonian: StackedHamiltonian[_BC0Inv],
) -> Hamiltonian[_BC0Inv]:
    n_states = np.prod(hamiltonian["array"].shape[:3])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(n_states, n_states),
    }


def stack_hamiltonian(hamiltonian: Hamiltonian[_BC0Inv]) -> StackedHamiltonian[_BC0Inv]:
    basis = BasisConfigUtil[Any, Any, Any](hamiltonian["basis"])
    return {
        "basis": hamiltonian["basis"],
        "array": hamiltonian["array"].reshape(*basis.shape, *basis.shape),
    }


@overload
def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonianWithBasis[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    n: _L0Inv,
    axis: Literal[0, -3],
) -> StackedHamiltonianWithBasis[TruncatedBasis[_L0Inv, _CBX0Inv], _CBX1Inv, _CBX2Inv]:
    ...


@overload
def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonianWithBasis[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    n: _L0Inv,
    axis: Literal[1, -2],
) -> StackedHamiltonianWithBasis[_CBX0Inv, TruncatedBasis[_L0Inv, _CBX1Inv], _CBX2Inv]:
    ...


@overload
def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonianWithBasis[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    n: _L0Inv,
    axis: Literal[2, -1],
) -> StackedHamiltonianWithBasis[_CBX0Inv, _CBX1Inv, TruncatedBasis[_L0Inv, _CBX2Inv]]:
    ...


def truncate_hamiltonian_basis(
    hamiltonian: StackedHamiltonianWithBasis[_CBX0Inv, _CBX1Inv, _CBX2Inv],
    n: _L0Inv,
    axis: Literal[0, 1, 2, -1, -2, -3] = -1,
) -> (
    StackedHamiltonianWithBasis[TruncatedBasis[_L0Inv, _CBX0Inv], _CBX1Inv, _CBX2Inv]
    | StackedHamiltonianWithBasis[_CBX0Inv, TruncatedBasis[_L0Inv, _CBX1Inv], _CBX2Inv]
    | StackedHamiltonianWithBasis[_CBX0Inv, _CBX1Inv, TruncatedBasis[_L0Inv, _CBX2Inv]]
):
    parent_basis: FundamentalBasis[Any] = hamiltonian["basis"][axis % 3]
    if not is_basis_type(parent_basis, "momentum"):
        raise NotImplementedError
    padded = pad_ft_points(
        hamiltonian["array"], s=(n, n), axes=(axis % 3, 3 + (axis % 3))
    )
    new_basis: TruncatedBasis[Any, Any] = {
        "n": n,
        "_type": "truncated",
        "parent": parent_basis,
    }
    basis: list[Basis[Any, Any]] = list(hamiltonian["basis"])
    basis[axis] = new_basis
    return {"array": padded, "basis": tuple(basis)}  # type: ignore[]


_BX = TypeVar("_BX", bound=Basis[Any, Any], contravariant=True)


def add_hamiltonian(
    a: Hamiltonian[_BC0Inv], b: Hamiltonian[_BC0Inv]
) -> Hamiltonian[_BC0Inv]:
    """
    Add together two operators.

    Parameters
    ----------
    a : Hamiltonian[_BC0Inv]
    b : Hamiltonian[_BC0Inv]

    Returns
    -------
    Hamiltonian[_BC0Inv]
    """
    return {"basis": a["basis"], "array": a["array"] + b["array"]}


def _convert_explicit_basis_x2(
    hamiltonian: _StackedHamiltonianPoints[_L0Inv, _L1Inv, _L2Inv],
    basis: np.ndarray[tuple[_LInv, _L2Inv], np.dtype[np.complex_]],
) -> _StackedHamiltonianPoints[_L0Inv, _L1Inv, _LInv]:
    end_dot = np.sum(
        hamiltonian[slice_along_axis(np.newaxis, -2)]
        * basis.reshape(1, 1, 1, 1, 1, *basis.shape),
        axis=-1,
    )
    return np.sum(  # type: ignore[no-any-return]
        end_dot[slice_along_axis(np.newaxis, 2)]
        * basis.conj().reshape(1, 1, *basis.shape, 1, 1, 1),
        axis=3,
    )


def convert_x2_to_explicit_basis(
    hamiltonian: HamiltonianWithBasis[_BX0Inv, _BX1Inv, MomentumBasis[_L0Inv]],
    basis: ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]],
) -> HamiltonianWithBasis[
    _BX0Inv, _BX1Inv, ExplicitBasis[_L1Inv, PositionBasis[_L0Inv]]
]:
    stacked = stack_hamiltonian(hamiltonian)

    x2_position = np.fft.fftn(
        np.fft.ifftn(stacked["array"], axes=(2,), norm="ortho"),
        axes=(5,),
        norm="ortho",
    )
    x2_explicit = _convert_explicit_basis_x2(x2_position, basis["vectors"])

    return flatten_hamiltonian(
        {
            "basis": (hamiltonian["basis"][0], hamiltonian["basis"][1], basis),
            "array": x2_explicit,  # type: ignore[typeddict-item]
        }
    )
