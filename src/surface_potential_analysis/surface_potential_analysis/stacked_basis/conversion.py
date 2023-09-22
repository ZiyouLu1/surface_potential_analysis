from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from surface_potential_analysis.axis.conversion import (
    axis_as_fundamental_axis,
    axis_as_fundamental_momentum_axis,
    axis_as_fundamental_position_axis,
    axis_as_fundamental_transformed_axis,
    axis_as_n_point_axis,
)
from surface_potential_analysis.axis.stacked_axis import (
    StackedBasis,
    StackedBasisLike,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalBasis,
        FundamentalPositionBasis,
        FundamentalTransformedBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.axis.axis_like import (
        BasisLike,
        BasisWithLengthLike,
    )

    _BL0 = TypeVar("_BL0", bound=BasisWithLengthLike[Any, Any, Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def stacked_basis_as_fundamental_transformed_basis(
    basis: StackedBasisLike[*tuple[_B0, ...]],
) -> StackedBasisLike[*tuple[FundamentalTransformedBasis[Any], ...]]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumAxis[Any, Any], ...]
    """
    return StackedBasis(
        *tuple(axis_as_fundamental_transformed_axis(axis) for axis in basis)
    )


def stacked_basis_as_fundamental_basis(
    basis: StackedBasisLike[*tuple[_B0, ...]],
) -> StackedBasisLike[*tuple[FundamentalBasis[Any], ...]]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumAxis[Any, Any], ...]
    """
    return StackedBasis(*tuple(axis_as_fundamental_axis(axis) for axis in basis))


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisLike[_BL0],
) -> StackedBasisLike[FundamentalTransformedPositionBasis[Any, Literal[1]]]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisLike[_BL0, _BL0],
) -> StackedBasisLike[
    FundamentalTransformedPositionBasis[Any, Literal[2]],
    FundamentalTransformedPositionBasis[Any, Literal[2]],
]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisLike[_BL0, _BL0, _BL0],
) -> StackedBasisLike[
    FundamentalTransformedPositionBasis[Any, Literal[3]],
    FundamentalTransformedPositionBasis[Any, Literal[3]],
    FundamentalTransformedPositionBasis[Any, Literal[3]],
]:
    ...


@overload
def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisLike[*tuple[_BL0, ...]],
) -> StackedBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]:
    ...


def stacked_basis_as_fundamental_momentum_basis(
    basis: StackedBasisLike[*tuple[_BL0, ...]]
    | StackedBasisLike[_BL0]
    | StackedBasisLike[_BL0, _BL0]
    | StackedBasisLike[_BL0, _BL0, _BL0],
) -> StackedBasisLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]:
    """
    Get the fundamental momentum basis for a given basis.

    Parameters
    ----------
    basis : _ALB0Inv

    Returns
    -------
    tuple[FundamentalMomentumAxis[Any, Any], ...]
    """
    return StackedBasis(
        *tuple(axis_as_fundamental_momentum_axis(axis) for axis in basis)
    )


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisLike[_BL0],
) -> StackedBasisLike[FundamentalPositionBasis[Any, Literal[1]]]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisLike[_BL0, _BL0],
) -> StackedBasisLike[
    FundamentalPositionBasis[Any, Literal[2]],
    FundamentalPositionBasis[Any, Literal[2]],
]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisLike[_BL0, _BL0, _BL0],
) -> StackedBasisLike[
    FundamentalPositionBasis[Any, Literal[3]],
    FundamentalPositionBasis[Any, Literal[3]],
    FundamentalPositionBasis[Any, Literal[3]],
]:
    ...


@overload
def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisLike[*tuple[Any, ...]],
) -> StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    ...


def stacked_basis_as_fundamental_position_basis(
    basis: StackedBasisLike[*tuple[_BL0, ...]]
    | StackedBasisLike[_BL0]
    | StackedBasisLike[_BL0, _BL0]
    | StackedBasisLike[_BL0, _BL0, _BL0],
) -> StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    """
    Get the fundamental position basis for a given basis.

    Parameters
    ----------
    self : BasisConfigUtil[tuple[BasisLike[_LF0Inv, _L0Inv], BasisLike[_LF1Inv, _L1Inv], BasisLike[_LF2Inv, _L2Inv]]]

    Returns
    -------
    StackedAxisLike[tuple[FundamentalPositionBasis[_LF0Inv], FundamentalPositionBasis[_LF1Inv], FundamentalPositionBasis[_LF2Inv]]
    """
    return StackedBasis(
        *tuple(axis_as_fundamental_position_axis(axis) for axis in basis)
    )


def stacked_basis_as_fundamental_with_shape(
    basis: StackedBasisLike[*tuple[_BL0, ...]],
    shape: tuple[int, ...],
) -> StackedBasisLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]:
    """
    Given a basis get a fundamental position basis with the given shape.

    Parameters
    ----------
    basis : Basis[_NDInv]
    shape : tuple[int, ...]

    Returns
    -------
    Basis[_NDInv]
    """
    return StackedBasis(
        *tuple(
            axis_as_n_point_axis(ax, n=n) for (ax, n) in zip(basis, shape, strict=True)
        )
    )
