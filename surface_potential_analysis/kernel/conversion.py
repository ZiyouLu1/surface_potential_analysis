from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.operator.conversion import (
    convert_operator_list_to_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.kernel.kernel import NoiseOperatorList

    _B0Inv = TypeVar("_B0Inv", bound=BasisLike[Any, Any])
    _B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
    _B2Inv = TypeVar("_B2Inv", bound=BasisLike[Any, Any])
    _B3Inv = TypeVar("_B3Inv", bound=BasisLike[Any, Any])
    _B4 = TypeVar("_B4", bound=BasisLike[Any, Any])


def convert_noise_operator_list_to_basis(
    operator: NoiseOperatorList[_B4, _B0Inv, _B1Inv],
    basis: StackedBasisLike[_B2Inv, _B3Inv],
) -> NoiseOperatorList[_B4, _B2Inv, _B3Inv]:
    """Given a noise operator, convert it to the given basis.

    Parameters
    ----------
    operator : NoiseOperatorList[_B4, _B0Inv, _B1Inv]
    basis : StackedBasisLike[_B2Inv, _B3Inv]

    Returns
    -------
    NoiseOperatorList[_B4, _B2Inv, _B3Inv]
    """
    converted = convert_operator_list_to_basis(operator, basis)
    return {
        "basis": converted["basis"],
        "data": converted["data"],
        "eigenvalue": operator["eigenvalue"],
    }
