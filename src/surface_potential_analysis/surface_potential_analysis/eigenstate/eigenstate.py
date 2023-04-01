from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import (
    Basis,
    ExplicitBasis,
    FundamentalBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
    PositionBasisConfig,
)
from surface_potential_analysis.interpolation import pad_ft_points

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])


class Eigenstate(TypedDict, Generic[_BC0Cov]):
    basis: _BC0Cov
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


EigenstateWithBasis = Eigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def save_eigenstate(path: Path, eigenstates: Eigenstate[Any]) -> None:
    state = np.array(eigenstates, dtype=Eigenstate)
    np.save(path, state)


def load_eigenstate(path: Path) -> Eigenstate[Any]:
    return np.load(path)[()]  # type:ignore


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)

PositionBasisEigenstate = Eigenstate[PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]]

MomentumBasisEigenstate = Eigenstate[MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]]


class EigenstateList(TypedDict, Generic[_BC0Cov]):
    """
    Represents a list of eigenstates, each with the same basis
    and bloch wavevector
    """

    basis: _BC0Cov
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int], np.dtype[np.float_]]


def save_eigenstate_list(path: Path, eigenstates: EigenstateList[Any]) -> None:
    state = np.array(eigenstates, dtype=EigenstateList)
    np.save(path, state)


def load_eigenstate_list(path: Path) -> EigenstateList[Any]:
    return np.load(path)[()]  # type:ignore


_L0Inv = TypeVar("_L0Inv", bound=int, covariant=True)
_L1Inv = TypeVar("_L1Inv", bound=int, covariant=True)
_L2Inv = TypeVar("_L2Inv", bound=int, covariant=True)


def convert_eigenstate_to_momentum_basis(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    raise NotImplementedError()


def convert_eigenstate_to_position_basis(
    eigenstate: MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    raise NotImplementedError()


class StackedEigenstate(TypedDict, Generic[_BC0Cov]):
    basis: _BC0Cov
    vector: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]


StackedEigenstateWithBasis = StackedEigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def stack_eigenstate(state: Eigenstate[_BC0Inv]) -> StackedEigenstate[_BC0Inv]:
    util = BasisConfigUtil(state["basis"])
    return {"basis": state["basis"], "vector": state["vector"].reshape(util.shape)}


def flatten_eigenstate(state: StackedEigenstate[_BC0Inv]) -> Eigenstate[_BC0Inv]:
    return {"basis": state["basis"], "vector": state["vector"].reshape(-1)}


def _convert_momentum_basis_x01_to_position(
    eigenstate: StackedEigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]] | MomentumBasis[_L0Inv],
        _BX0Inv,
    ]
) -> StackedEigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]:
    basis = BasisConfigUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"],
        s=[basis.fundamental_n0, basis.fundamental_n1],
        axes=(0, 1),
    )
    transformed = np.fft.ifftn(padded, axes=(0, 1), norm="forward")  # TODO: fix this??
    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": basis.delta_x0,
                "n": basis.fundamental_n0,  # type: ignore
            },
            {
                "_type": "position",
                "delta_x": basis.delta_x1,
                "n": basis.fundamental_n1,  # type: ignore
            },
            eigenstate["basis"][2],
        ),
        "vector": transformed,
    }


_PInv = TypeVar("_PInv", bound=FundamentalBasis[Any])


def _convert_explicit_basis_x2_to_position(
    eigenstate: StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, ExplicitBasis[Any, _PInv]]
) -> StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, _PInv]:
    vector = np.sum(
        eigenstate["vector"][:, :, :, np.newaxis]
        * eigenstate["basis"][2]["vectors"][np.newaxis, np.newaxis, :, :],
        axis=2,
    )
    return {
        "basis": (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            eigenstate["basis"][2]["parent"],
        ),
        "vector": vector,
    }


def convert_sho_eigenstate_to_position_basis(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]],
        ExplicitBasis[Any, PositionBasis[_L2Inv]],
    ]
) -> PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    stacked = stack_eigenstate(eigenstate)
    xy_converted = _convert_momentum_basis_x01_to_position(stacked)
    converted = _convert_explicit_basis_x2_to_position(xy_converted)
    return flatten_eigenstate(converted)
