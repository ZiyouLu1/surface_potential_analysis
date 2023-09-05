from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalAxis,
    FundamentalPositionAxis,
    FundamentalPositionAxis3d,
    FundamentalTransformedPositionAxis3d,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
        FundamentalPositionBasis3d,
    )

    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _S1Inv = TypeVar("_S1Inv", bound=tuple[int, int])


_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


def position_basis_3d_from_parent(
    parent: _B3d0Inv, resolution: tuple[_NF0Inv, _NF1Inv, _NF2Inv]
) -> FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given a parent basis construct another basis with the same lattice vectors.

    Parameters
    ----------
    parent : _B3d0Inv
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]

    Returns
    -------
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
        _description_
    """
    return (
        FundamentalPositionAxis3d(parent[0].delta_x, resolution[0]),
        FundamentalPositionAxis3d(parent[1].delta_x, resolution[1]),
        FundamentalPositionAxis3d(parent[2].delta_x, resolution[2]),
    )


def fundamental_basis_from_shape(
    shape: _S0Inv,
) -> tuple[FundamentalAxis[int], ...]:
    """
    Given a resolution and a set of directions construct a FundamentalPositionBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    return tuple(FundamentalAxis(n) for n in shape)


@overload
def position_basis_from_shape(
    shape: _S0Inv,
) -> tuple[FundamentalPositionAxis[Any, Any], ...]:
    ...


@overload
def position_basis_from_shape(
    shape: _S0Inv,
    delta_x: tuple[np.ndarray[_S1Inv, np.dtype[np.float_]]] | None = None,
) -> tuple[FundamentalPositionAxis[Any, int], ...]:
    ...


def position_basis_from_shape(
    shape: _S0Inv,
    delta_x: tuple[np.ndarray[_S1Inv, np.dtype[np.float_]]] | None = None,
) -> tuple[FundamentalPositionAxis[Any, int], ...]:
    """
    Given a resolution and a set of directions construct a FundamentalPositionBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalPositionBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    delta_x = np.eye(len(shape)) if delta_x is None else delta_x
    return tuple(
        FundamentalPositionAxis(dx, n) for dx, n in zip(delta_x, shape, strict=True)
    )


def position_basis_3d_from_shape(
    shape: tuple[_NF0Inv, _NF1Inv, _NF2Inv],
    delta_x: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]
    | None = None,
) -> tuple[
    FundamentalPositionAxis[_NF0Inv, Literal[3]],
    FundamentalPositionAxis[_NF1Inv, Literal[3]],
    FundamentalPositionAxis[_NF2Inv, Literal[3]],
]:
    """
    Given a shape generate a 3d position basis.

    Parameters
    ----------
    shape : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
    delta_x : np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]] | None, optional
        delta_x as list of individual delta_x, by default None

    Returns
    -------
    tuple[FundamentalPositionAxis[_NF0Inv, Literal[3]], FundamentalPositionAxis[_NF1Inv, Literal[3]], FundamentalPositionAxis[_NF2Inv, Literal[3]]]
    """
    return position_basis_from_shape(shape, delta_x)  # type: ignore[arg-type,return-value]


def momentum_basis_3d_from_resolution(
    resolution: tuple[_NF0Inv, _NF1Inv, _NF2Inv],
    delta_x: tuple[
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
        np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    ]
    | None = None,
) -> FundamentalMomentumBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]:
    """
    Given a resolution and a set of directions construct a FundamentalMomentumBasisConfig.

    Parameters
    ----------
    resolution : tuple[_NF0Inv, _NF1Inv, _NF2Inv]
        resolution of the basis
    delta_x : tuple[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], np.ndarray[tuple[Literal[3]], np.dtype[np.float_]], ] | None, optional
        vectors for the basis, by default (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))

    Returns
    -------
    FundamentalMomentumBasis3d[_NF0Inv, _NF1Inv, _NF2Inv]
    """
    delta_x = (
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        if delta_x is None
        else delta_x
    )
    return (
        FundamentalTransformedPositionAxis3d(delta_x[0], resolution[0]),
        FundamentalTransformedPositionAxis3d(delta_x[1], resolution[1]),
        FundamentalTransformedPositionAxis3d(delta_x[2], resolution[2]),
    )
