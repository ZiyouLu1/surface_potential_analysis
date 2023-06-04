from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.conversion import convert_vector

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import Basis
    from surface_potential_analysis.potential.potential import Potential

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _B1Inv = TypeVar("_B1Inv", bound=Basis[Any])


def convert_potential_to_basis(
    potential: Potential[_B0Inv], basis: _B1Inv
) -> Potential[_B1Inv]:
    """
    Given an potential, calculate the potential in the given basis.

    Parameters
    ----------
    potential : Potential[_B0Inv]
    basis : _B1Inv

    Returns
    -------
    Potential[_B1Inv]
    """
    converted = convert_vector(potential["vector"], potential["basis"], basis)
    return {"basis": basis, "vector": converted}  # type: ignore[typeddict-item]
