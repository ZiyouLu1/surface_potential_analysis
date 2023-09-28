from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.operator.operator_list import OperatorList

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import (
        DiagonalOperator,
        SingleBasisDiagonalOperator,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketList,
        WavepacketWithEigenvaluesList,
    )

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

LocalizationOperator = OperatorList[_B0, _B1, _B2]
"""A list of operators, acting on each bloch k"""

_SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
_SB1 = TypeVar("_SB1", bound=StackedBasisLike[*tuple[Any, ...]])


def get_localized_wavepackets(
    wavepackets: WavepacketList[_B2, _SB1, _SB0],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> WavepacketList[_B1, _SB1, _SB0]:
    """
    Apply the LocalizationOperator to produce localized wavepackets.

    Parameters
    ----------
    wavepackets : WavepacketList[_B2, _SB1, _SB0]
        The unlocalized wavepackets
    operator : LocalizationOperator[_SB1, _B1, _B2]
        The operator used to localize the wavepackets

    Returns
    -------
    WavepacketList[_B1, _SB1, _SB0]
        The localized wavepackets
    """
    stacked_operator = operator["data"].reshape(-1, *operator["basis"][1].shape)
    coefficients = np.moveaxis(stacked_operator, 0, -1)
    vectors = wavepackets["data"].reshape(*wavepackets["basis"][0].shape, -1)

    data = np.sum(
        coefficients[:, :, :, np.newaxis] * vectors[np.newaxis, :, :, :], axis=(1)
    )
    return {
        "basis": StackedBasis(
            StackedBasis(operator["basis"][1][0], wavepackets["basis"][0][1]),
            wavepackets["basis"][1],
        ),
        "data": data.reshape(-1),
    }


def get_wavepacket_hamiltonian(
    wavepackets: WavepacketWithEigenvaluesList[_B0, _SB1, _SB0]
) -> DiagonalOperator[StackedBasisLike[_B0, _SB1], StackedBasisLike[_B0, _SB1]]:
    """
    Get the Hamiltonian in the Wavepacket basis.

    Parameters
    ----------
    wavepackets : WavepacketWithEigenvaluesList[_B0, _SB1, _SB0]

    Returns
    -------
    DiagonalOperator[StackedBasisLike[_B0, _SB1], StackedBasisLike[_B0, _SB1]]
    """
    return {
        "basis": StackedBasis(wavepackets["basis"][0], wavepackets["basis"][0]),
        "data": wavepackets["eigenvalue"],
    }


def get_localized_hamiltonian_from_eigenvalues(
    hamiltonian: SingleBasisDiagonalOperator[StackedBasisLike[_B2, _SB1],],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> OperatorList[_SB1, _B1, _B1]:
    hamiltonian_stacked = hamiltonian["data"].reshape(*hamiltonian["basis"][0].shape)
    operator_stacked = operator["data"].reshape(-1, *operator["basis"][1].shape)

    # Hamiltonian n, k; is diagonal
    # Operator k n'; n with k just along diagonal
    # Move k to end, and do tensordot on n and k. Since H is diagonal we dont need to sum
    converted_front = (
        np.moveaxis(operator_stacked, 0, -1)[:, :, :]
        * hamiltonian_stacked[np.newaxis, :, :]
    )

    # converted_front is n'; n, k
    # Operator k n'; n move axis to get n'; n k and conj to make dual basis
    # this time we sum over the n axis
    converted = np.sum(
        converted_front[:, np.newaxis, :, :]
        * np.conj(np.moveaxis(operator_stacked, 0, -1))[np.newaxis, :, :, :],
        axis=(2),
    )
    # we now have n'; n' k, so we need to move k to the front
    # this is why we return an operator list
    return {
        "basis": StackedBasis(
            hamiltonian["basis"][0][1],
            StackedBasis(operator["basis"][1][0], operator["basis"][1][0]),
        ),
        "data": np.moveaxis(converted, -1, 0).reshape(-1),
    }


def get_localized_wavepacket_hamiltonian(
    wavepackets: WavepacketWithEigenvaluesList[_B2, _SB1, _SB0],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> OperatorList[_SB1, _B1, _B1]:
    hamiltonian = get_wavepacket_hamiltonian(wavepackets)
    return get_localized_hamiltonian_from_eigenvalues(hamiltonian, operator)
