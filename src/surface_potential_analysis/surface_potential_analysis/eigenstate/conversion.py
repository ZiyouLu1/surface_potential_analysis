from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.basis import (
    Basis,
    BasisUtil,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
    as_fundamental_basis,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.basis_config.conversion import convert_vector
from surface_potential_analysis.util.interpolation import pad_ft_points
from surface_potential_analysis.util.util import slice_along_axis

if TYPE_CHECKING:
    from surface_potential_analysis._types import SingleIndexLike

    from .eigenstate import (
        Eigenstate,
        EigenstateWithBasis,
        MomentumBasisEigenstate,
        PositionBasisEigenstate,
    )

    _BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
    _BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])

    _BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
    _BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
    _BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

    _L0Inv = TypeVar("_L0Inv", bound=int, covariant=True)
    _L1Inv = TypeVar("_L1Inv", bound=int, covariant=True)
    _L2Inv = TypeVar("_L2Inv", bound=int, covariant=True)

    _LF0Inv = TypeVar("_LF0Inv", bound=int)
    _LF1Inv = TypeVar("_LF1Inv", bound=int)


def convert_eigenstate_to_basis(
    eigenstate: Eigenstate[_BC0Inv], basis: _BC1Inv
) -> Eigenstate[_BC1Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    converted = convert_vector(eigenstate["vector"], eigenstate["basis"], basis)
    return {"basis": basis, "vector": converted}  # type: ignore[typeddict-item]


def convert_eigenstate_to_position_basis(
    eigenstate: Eigenstate[_BC0Inv],
) -> EigenstateWithBasis[PositionBasis[Any], PositionBasis[Any], PositionBasis[Any]]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    util = BasisConfigUtil(eigenstate["basis"])
    return convert_eigenstate_to_basis(
        eigenstate, util.get_fundamental_basis_in("position")
    )


def convert_eigenstate_to_momentum_basis(
    eigenstate: Eigenstate[_BC0Inv],
) -> EigenstateWithBasis[MomentumBasis[Any], MomentumBasis[Any], MomentumBasis[Any]]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    util = BasisConfigUtil(eigenstate["basis"])
    return convert_eigenstate_to_basis(
        eigenstate, util.get_fundamental_basis_in("momentum")
    )


@overload
def flaten_eigenstate_x(
    eigenstate: EigenstateWithBasis[_BX0Inv, _BX1Inv, _BX2Inv],
    idx: SingleIndexLike,
    z_axis: Literal[0, -3],
) -> EigenstateWithBasis[PositionBasis[Literal[1]], _BX1Inv, _BX2Inv]:
    ...


@overload
def flaten_eigenstate_x(
    eigenstate: EigenstateWithBasis[_BX0Inv, _BX1Inv, _BX2Inv],
    idx: SingleIndexLike,
    z_axis: Literal[1, -2],
) -> EigenstateWithBasis[_BX0Inv, PositionBasis[Literal[1]], _BX2Inv]:
    ...


@overload
def flaten_eigenstate_x(
    eigenstate: EigenstateWithBasis[_BX0Inv, _BX1Inv, _BX2Inv],
    idx: SingleIndexLike,
    z_axis: Literal[2, -1],
) -> EigenstateWithBasis[_BX0Inv, _BX1Inv, PositionBasis[Literal[1]]]:
    ...


def flaten_eigenstate_x(
    eigenstate: EigenstateWithBasis[_BX0Inv, _BX1Inv, _BX2Inv],
    idx: SingleIndexLike,
    z_axis: Literal[0, 1, 2, -1, -2, -3],
) -> EigenstateWithBasis[Any, Any, Any]:
    """
    Flatten the eigenstate in the z direction, at the given index in position basis.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[_BX0Inv, _BX1Inv, _BX2Inv]
    idx : int
        index in position basis to flatten
    z_axis : Literal[0, 1, 2, -1, -2, -3]
        axis along which to flatten

    Returns
    -------
    EigenstateWithBasis[Any, Any, Any]
        _description_
    """
    position_basis = (
        eigenstate["basis"][0],
        eigenstate["basis"][1],
        BasisUtil(eigenstate["basis"][2]).get_fundamental_basis_in("position"),
    )
    util = BasisConfigUtil(position_basis)
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    converted = convert_eigenstate_to_basis(eigenstate, position_basis)
    flattened = (
        converted["vector"]
        .reshape(*util.shape)[slice_along_axis(idx, z_axis)]
        .reshape(-1)
    )
    position_basis[2]["n"] = 1
    return {"basis": position_basis, "vector": flattened}


def convert_position_basis_eigenstate_to_momentum_basis(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    """
    convert an eigenstate from position to momentum basis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
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


def convert_momentum_basis_eigenstate_to_position_basis(
    eigenstate: MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert an eigenstate from momentum to position basis.

    Parameters
    ----------
    eigenstate : MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
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


@overload
def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    ...


@overload
def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    ...


@overload
def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    ...


def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
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
