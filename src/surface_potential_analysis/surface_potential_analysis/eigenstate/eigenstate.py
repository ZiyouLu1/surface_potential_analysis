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
from surface_potential_analysis.basis.basis import BasisUtil, as_fundamental_basis
from surface_potential_analysis.basis_config.basis_config import (
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
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])


class Eigenstate(TypedDict, Generic[_BC0Cov]):
    """represents an eigenstate in an explicit basis."""

    basis: _BC0Cov
    vector: np.ndarray[tuple[int], np.dtype[np.complex_]]


EigenstateWithBasis = Eigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def save_eigenstate(path: Path, eigenstates: Eigenstate[Any]) -> None:
    """
    Save an eigenstate in an npy file.

    Parameters
    ----------
    path : Path
    eigenstates : Eigenstate[Any]
    """
    state = np.array(eigenstates, dtype=Eigenstate)
    np.save(path, state)


def load_eigenstate(path: Path) -> Eigenstate[Any]:
    """
    Load an eigenstate from an npy file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Eigenstate[Any]
    """
    return np.load(path)[()]  # type:ignore[no-any-return]


_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)


PositionBasisEigenstate = Eigenstate[PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]]

MomentumBasisEigenstate = Eigenstate[MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]]


class EigenstateList(TypedDict, Generic[_BC0Cov]):
    """Represents a list of eigenstates, each with the same basis and bloch wavevector."""

    basis: _BC0Cov
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[int], np.dtype[np.float_]]


def save_eigenstate_list(path: Path, eigenstates: EigenstateList[Any]) -> None:
    state = np.array(eigenstates, dtype=EigenstateList)
    np.save(path, state)


def load_eigenstate_list(path: Path) -> EigenstateList[Any]:
    return np.load(path)[()]  # type:ignore[no-any-return]


_L0Inv = TypeVar("_L0Inv", bound=int, covariant=True)
_L1Inv = TypeVar("_L1Inv", bound=int, covariant=True)
_L2Inv = TypeVar("_L2Inv", bound=int, covariant=True)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)


def convert_eigenstate_to_momentum_basis(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    util = BasisConfigUtil(eigenstate["basis"])
    transformed = np.fft.fftn(
        eigenstate["vector"].reshape(util.shape),
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": (
            {
                "_type": "momentum",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "momentum",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            {
                "_type": "momentum",
                "delta_x": util.delta_x2,
                "n": util.fundamental_n2,  # type: ignore[typeddict-item]
            },
        ),
        "vector": transformed.reshape(-1),
    }


def convert_eigenstate_to_position_basis(
    eigenstate: MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    util = BasisConfigUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"].reshape(util.shape),
        s=util.fundamental_shape,
        axes=(0, 1, 2),
    )
    transformed = np.fft.ifftn(
        padded,
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x2,
                "n": util.fundamental_n2,  # type: ignore[typeddict-item]
            },
        ),
        "vector": transformed.reshape(-1),
    }


class _StackedEigenstate(TypedDict, Generic[_BC0Cov]):
    basis: _BC0Cov
    vector: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]


StackedEigenstateWithBasis = _StackedEigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def _stack_eigenstate(
    state: Eigenstate[BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]
) -> _StackedEigenstate[BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]]:
    util = BasisConfigUtil(state["basis"])
    return {"basis": state["basis"], "vector": state["vector"].reshape(util.shape)}


def _flatten_eigenstate(state: _StackedEigenstate[_BC0Inv]) -> Eigenstate[_BC0Inv]:
    return {"basis": state["basis"], "vector": state["vector"].reshape(-1)}


def _convert_momentum_basis_x01_to_position(
    eigenstate: StackedEigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> StackedEigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]:
    util = BasisConfigUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"],
        s=[util.fundamental_n0, util.fundamental_n1],
        axes=(0, 1),
    )
    transformed = np.fft.ifftn(padded, axes=(0, 1), norm="ortho")
    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            eigenstate["basis"][2],
        ),
        "vector": transformed,
    }


def _convert_position_basis_x2_to_momentum(
    eigenstate: StackedEigenstateWithBasis[
        _BX0Inv,
        _BX1Inv,
        TruncatedBasis[_L0Inv, PositionBasis[_LF0Inv]],
    ]
    | StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, PositionBasis[_LF0Inv]],
) -> StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, MomentumBasis[_LF0Inv]]:
    basis = BasisUtil(eigenstate["basis"][2])
    transformed = np.fft.fftn(
        eigenstate["vector"], axes=(2,), s=(basis.fundamental_n,), norm="ortho"
    )
    return {
        "basis": (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            {"_type": "momentum", "delta_x": basis.delta_x, "n": basis.fundamental_n},
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
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_LF0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_LF1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
) -> PositionBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]:
    stacked = _stack_eigenstate(eigenstate)
    xy_converted = _convert_momentum_basis_x01_to_position(stacked)
    converted = _convert_explicit_basis_x2_to_position(xy_converted)
    return _flatten_eigenstate(converted)


def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    """
    Given a truncated basis in xy, convert to a funadamental momentum basis of lower resolution.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ TruncatedBasis[Any, MomentumBasis[_L0Inv]]  |  MomentumBasis[_L0Inv], TruncatedBasis[Any, MomentumBasis[_L1Inv]]  |  MomentumBasis[_L1Inv], _BX0Inv, ]

    Returns
    -------
    EigenstateWithBasis[ TruncatedBasis[Any, MomentumBasis[_L0Inv]] | MomentumBasis[_L0Inv], TruncatedBasis[Any, MomentumBasis[_L1Inv]] | MomentumBasis[_L1Inv], _BX0Inv, ]
    """
    return {
        "basis": (
            as_fundamental_basis(eigenstate["basis"][0]),
            as_fundamental_basis(eigenstate["basis"][1]),
            eigenstate["basis"][2],
        ),
        "vector": eigenstate["vector"],
    }


def convert_sho_eigenstate_to_momentum_basis(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_LF0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_LF1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
) -> MomentumBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]:
    """
    Convert a sho eigenstate to momentum basis.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]]  |  MomentumBasis[_LF0Inv], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]]  |  MomentumBasis[_LF1Inv], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ]

    Returns
    -------
    MomentumBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]
    """
    stacked = _stack_eigenstate(eigenstate)
    x2_position = _convert_explicit_basis_x2_to_position(stacked)
    x2_momentum = _convert_position_basis_x2_to_momentum(x2_position)
    flattened = _flatten_eigenstate(x2_momentum)
    return convert_sho_eigenstate_to_fundamental_xy(flattened)


def convert_eigenstate_01_axis_to_position_basis(
    eigenstate: EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]:
    """
    convert eigenstate to position basis in the x0 and x1 direction.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv, ]

    Returns
    -------
    EigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]
    """
    stacked = _stack_eigenstate(eigenstate)
    converted = _convert_momentum_basis_x01_to_position(stacked)
    return _flatten_eigenstate(converted)


def calculate_normalisation(eigenstate: Eigenstate[Any]) -> float:
    """
    calculate the normalization of an eigenstate.

    This should always be 1

    Parameters
    ----------
    eigenstate: Eigenstate[Any]

    Returns
    -------
    float
    """
    return np.sum(np.conj(eigenstate["vector"]) * eigenstate["vector"])
