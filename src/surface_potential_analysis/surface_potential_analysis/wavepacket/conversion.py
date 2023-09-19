from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np

from surface_potential_analysis.axis.axis_like import (
    BasisLike,
    BasisWithLengthLike,
    convert_vector,
)
from surface_potential_analysis.axis.stacked_axis import StackedBasis
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.axis.stacked_axis import (
        StackedBasisLike,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        Wavepacket,
    )

    _B0Inv = TypeVar("_B0Inv", bound=StackedBasisLike[*tuple[Any, ...]])
    _B1Inv = TypeVar("_B1Inv", bound=StackedBasisLike[*tuple[Any, ...]])

    _B2Inv = TypeVar("_B2Inv", bound=StackedBasisLike[*tuple[Any, ...]])
    _B3Inv = TypeVar("_B3Inv", bound=StackedBasisLike[*tuple[Any, ...]])

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])


@overload
def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: _B1Inv,
    basis: None = None,
) -> Wavepacket[_B1Inv, _B2Inv]:
    ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: _B1Inv,
    basis: _B3Inv,
) -> Wavepacket[_B1Inv, _B3Inv]:
    ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: None = None,
    basis: _B3Inv,
) -> Wavepacket[_B0Inv, _B3Inv]:
    ...


@overload
def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: None = None,
    basis: None = None,
) -> Wavepacket[_B0Inv, _B2Inv]:
    ...


def convert_wavepacket_to_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: BasisLike[Any, Any] | None = None,
    basis: BasisLike[Any, Any] | None = None,
) -> Any:
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
    list_basis = wavepacket["basis"][0] if list_basis is None else list_basis
    basis = wavepacket["basis"][1] if basis is None else basis
    vectors = convert_vector(
        wavepacket["data"].reshape(wavepacket["basis"].shape),
        wavepacket["basis"][0],
        list_basis,
        axis=0,
    )
    vectors = convert_vector(vectors, wavepacket["basis"][1], basis, axis=1)
    return {"basis": StackedBasis(list_basis, basis), "data": vectors.reshape(-1)}


def convert_wavepacket_to_position_basis(
    wavepacket: Wavepacket[_B0Inv, StackedBasisLike[*tuple[_BL0, ...]]]
) -> Wavepacket[
    _B0Inv, StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, StackedAxisLike[tuple[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    stacked_basis_as_fundamental_position_basis(wavepacket["basis"][1])
    return convert_wavepacket_to_basis(
        wavepacket,
        basis=stacked_basis_as_fundamental_position_basis(wavepacket["basis"][1]),
    )


@overload
def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: _B1Inv,
) -> Wavepacket[
    _B1Inv,
    StackedBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
]:
    ...


@overload
def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: Wavepacket[_B0Inv, _B2Inv],
    *,
    list_basis: None = None,
) -> Wavepacket[
    _B0Inv,
    StackedBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]],
]:
    ...


def convert_wavepacket_to_fundamental_momentum_basis(
    wavepacket: Wavepacket[_B0Inv, StackedBasisLike[*tuple[_BL0, ...]]],
    *,
    list_basis: StackedBasisLike[*tuple[Any, ...]] | None = None,
) -> Any:
    """
    Convert a wavepacket to the fundamental position basis.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    Wavepacket[_NS0Inv, _NS1Inv, StackedAxisLike[tuple[PositionBasis[int], PositionBasis[int], PositionBasis[int]]]
    """
    return convert_wavepacket_to_basis(
        wavepacket,
        basis=stacked_basis_as_fundamental_momentum_basis(wavepacket["basis"][1]),
        list_basis=wavepacket["basis"][0] if list_basis is None else list_basis,
    )


def convert_wavepacket_to_shape(
    wavepacket: Wavepacket[_B0Inv, _B2Inv], shape: tuple[int, ...]
) -> Wavepacket[Any, _B2Inv]:
    """
    Convert the wavepacket to the given shape.

    Note that BasisUtil(wavepacket["list_basis"]).shape must be divisible by shape

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    shape : _S1Inv

    Returns
    -------
    Wavepacket[_S1Inv, _B0Inv]
    """
    old_shape = wavepacket["basis"][0].shape
    slices = tuple(
        slice(None, None, s0 // s1) for (s0, s1) in zip(old_shape, shape, strict=True)
    )
    np.testing.assert_array_almost_equal(
        old_shape,
        [s.step * s1 for (s, s1) in zip(slices, shape, strict=True)],
    )
    return {
        "basis": StackedBasis(
            fundamental_stacked_basis_from_shape(shape), wavepacket["basis"][1]
        ),
        "data": wavepacket["data"]
        .reshape(*old_shape, -1)[*slices, :]
        .reshape(np.prod(shape), -1),
    }
