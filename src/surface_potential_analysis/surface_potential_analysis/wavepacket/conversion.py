from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.build import fundamental_basis_from_shape
from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
    basis_as_fundamental_position_basis,
    convert_vector,
)
from surface_potential_analysis.basis.util import BasisUtil

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalPositionAxis,
        FundamentalTransformedPositionAxis,
    )
    from surface_potential_analysis.basis.basis import (
        AxisWithLengthBasis,
        Basis,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        Wavepacket,
    )

    _B0Inv = TypeVar("_B0Inv", bound=Basis)
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

    _B2Inv = TypeVar("_B2Inv", bound=AxisWithLengthBasis[Any])
    _B3Inv = TypeVar("_B3Inv", bound=AxisWithLengthBasis[Any])


def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    basis: _B3Inv,
) -> Wavepacket[_B0Inv, _B3Inv]:
    """
    Given a wavepacket convert it to the given basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv,  _B0Inv]
    basis : _B1Inv

    Returns
    -------
    Wavepacket[_S0Inv, _B1Inv]
    """
    vectors = convert_vector(wavepacket["vectors"], wavepacket["basis"], basis)
    return {
        "basis": basis,
        "shape": wavepacket["shape"],
        "vectors": vectors,  # type:ignore[typeddict-item]
    }


def convert_wavepacket_to_position_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv]
) -> Wavepacket[_B0Inv, tuple[FundamentalPositionAxis[Any, Any], ...]]:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, Basis3d[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    return convert_wavepacket_to_basis(
        wavepacket, basis_as_fundamental_position_basis(wavepacket["basis"])
    )


def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv]
) -> Wavepacket[_B0Inv, tuple[FundamentalTransformedPositionAxis[Any, Any], ...]]:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, Basis3d[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    return convert_wavepacket_to_basis(
        wavepacket, basis_as_fundamental_momentum_basis(wavepacket["basis"])
    )


def convert_wavepacket_to_shape(
    wavepacket: Wavepacket[_B0Inv, _B2Inv], shape: _S0Inv
) -> Wavepacket[Any, _B2Inv]:
    """
    Convert the wavepacket to the given shape.

    Note that wavepacket["shape"] must be divisible by shape

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    shape : _S1Inv

    Returns
    -------
    Wavepacket[_S1Inv, _B0Inv]
    """
    old_shape = BasisUtil(wavepacket["list_basis"]).shape
    slices = tuple(
        slice(None, None, s0 // s1) for (s0, s1) in zip(old_shape, shape, strict=True)
    )
    np.testing.assert_array_almost_equal(
        old_shape,
        [s.step * s1 for (s, s1) in zip(slices, shape, strict=True)],
    )
    return {
        "basis": wavepacket["basis"],
        "list_basis": fundamental_basis_from_shape(shape),
        "vectors": wavepacket["vectors"]
        .reshape(*old_shape, -1)[*slices, :]
        .reshape(np.prod(shape), -1),
    }
