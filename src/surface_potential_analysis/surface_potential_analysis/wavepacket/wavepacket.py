from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalAxis,
    FundamentalPositionAxis,
)
from surface_potential_analysis.basis.basis import (
    AxisWithLengthBasis,
    Basis,
    FundamentalMomentumBasis3d,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.basis.brillouin_zone import decrement_brillouin_zone
from surface_potential_analysis.basis.build import fundamental_basis_from_shape
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil, BasisUtil
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.eigenstate_collection import EigenstateList
from surface_potential_analysis.state_vector.state_vector_list import StateVectorList

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.operator.operator import SingleBasisOperator

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)

_B0Inv = TypeVar("_B0Inv", bound=Basis)
_B1Inv = TypeVar("_B1Inv", bound=AxisWithLengthBasis[Any])


_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


class Wavepacket(StateVectorList[_B0Inv, _B1Inv]):
    """represents an approximation of a Wannier function."""


class WavepacketWithEigenvalues(  # type: ignore[misc]
    EigenstateList[_B0Inv, _B1Inv], Wavepacket[_B0Inv, _B1Inv]
):
    """represents an approximation of a Wannier function."""


PositionBasisWavepacket3d = WavepacketWithEigenvalues[
    tuple[
        FundamentalAxis[_NS0Inv], FundamentalAxis[_NS1Inv], FundamentalAxis[Literal[1]]
    ],
    FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv],
]

MomentumBasisWavepacket3d = WavepacketWithEigenvalues[
    tuple[
        FundamentalAxis[_NS0Inv], FundamentalAxis[_NS1Inv], FundamentalAxis[Literal[1]]
    ],
    FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv],
]


def get_sample_basis(
    list_basis: _B0Inv,
    basis: AxisWithLengthBasis[_ND0Inv],
) -> AxisWithLengthBasis[_ND0Inv]:
    """
    Given the basis for a wavepacket, get the basis used to sample the packet.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    Basis[_ND0Inv]
    """
    return tuple(
        FundamentalPositionAxis(ax.delta_x * list_ax.n, list_ax.n)
        for (list_ax, ax) in zip(list_basis, basis, strict=True)
    )


def get_unfurled_basis(
    list_basis: _B0Inv,
    basis: AxisWithLengthBasis[_ND0Inv],
) -> AxisWithLengthBasis[_ND0Inv]:
    """
    Given the basis for a wavepacket, get the basis for the unfurled wavepacket.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    Basis[_ND0Inv]
    """
    return tuple(
        FundamentalPositionAxis(ax.delta_x * list_ax.n, ax.n * list_ax.n)
        for (list_ax, ax) in zip(list_basis, basis, strict=True)
    )


def get_furled_basis(
    basis: AxisWithLengthBasis[_ND0Inv], shape: _S0Inv
) -> AxisWithLengthBasis[_ND0Inv]:
    """
    Given the basis for an eigenstate, get the basis used for the furled wavepacket.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    Basis[_ND0Inv]
    """
    return tuple(
        FundamentalPositionAxis(ax.delta_x // n, ax.n // n)
        for (ax, n) in zip(basis, shape, strict=True)
    )


def get_wavepacket_sample_fractions(
    list_basis: _B0Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
    """
    Get the frequencies of the samples in a wavepacket, as a fraction of dk.

    Parameters
    ----------
    shape : np.ndarray[tuple[_NDInv], np.dtype[np.int_]]

    Returns
    -------
    np.ndarray[tuple[Literal[_NDInv], int], np.dtype[np.float_]]
    """
    util = BasisUtil(list_basis)
    return util.fundamental_nk_points / np.array(util.shape)[:, np.newaxis]  # type: ignore[no-any-return]


def get_wavepacket_sample_frequencies(
    list_basis: _B0Inv, basis: AxisWithLengthBasis[_ND0Inv]
) -> np.ndarray[tuple[_ND0Inv, int], np.dtype[np.float_]]:
    """
    Get the frequencies used in a given wavepacket.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : tuple length _ND0Inv

    Returns
    -------
    np.ndarray[tuple[_ND0Inv, int], np.dtype[np.float_]]
    """
    sample_basis = get_sample_basis(list_basis, basis)
    util = AxisWithLengthBasisUtil(sample_basis)
    return util.fundamental_k_points  # type: ignore[return-value]


def get_wavepacket_sample_frequencies_first_brillouin_zone(
    list_basis: _B0Inv, basis: AxisWithLengthBasis[_ND0Inv]
) -> np.ndarray[tuple[_ND0Inv, int], np.dtype[np.float_]]:
    """
    Get the frequencies used in the wavepacket wrapped to the first brillouin zone.

    Parameters
    ----------
    basis : Basis[_ND0Inv]
    shape : _S0Inv

    Returns
    -------
    np.ndarray[tuple[_ND0Inv, int], np.dtype[np.float_]]
    """
    sample_basis = get_sample_basis(list_basis, basis)
    util = AxisWithLengthBasisUtil(sample_basis)
    nk_points = util.nk_points
    for _ in util.shape:
        nk_points = decrement_brillouin_zone(basis, nk_points)
    return util.get_k_points_at_index(nk_points)  # type: ignore[return-value]


def generate_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_ND0Inv], np.dtype[np.float_]]],
        SingleBasisOperator[_B1Inv],
    ],
    shape: _S0Inv,
    *,
    save_bands: np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
) -> list[WavepacketWithEigenvalues[Any, _B1Inv]]:
    """
    Generate a wavepacket with the given number of samples.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_B3d0Inv]]
    shape : _S0Inv
    save_bands : np.ndarray[tuple[int], np.dtype[np.int_]] | None, optional

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]]]
    """
    list_basis = fundamental_basis_from_shape(shape)
    bloch_fractions = get_wavepacket_sample_fractions(list_basis)
    h = hamiltonian_generator(bloch_fractions[:, 0])
    basis_size = AxisWithLengthBasisUtil(h["basis"]).size
    save_bands = np.array([0]) if save_bands is None else save_bands
    subset_by_index: tuple[int, int] = (np.min(save_bands), np.max(save_bands))

    n_samples = np.prod(shape)
    vectors = np.empty((n_samples, basis_size), dtype=np.complex128)
    energies = np.empty(n_samples, dtype=np.float_)
    out: list[WavepacketWithEigenvalues[Any, _B1Inv]] = [
        {
            "basis": h["basis"],
            "vectors": vectors.copy(),
            "eigenvalues": energies.copy(),
            "list_basis": list_basis,
        }
        for _ in save_bands
    ]

    for i in range(np.prod(shape)):
        h = hamiltonian_generator(bloch_fractions[:, i])
        eigenstates = calculate_eigenvectors_hermitian(h, subset_by_index)

        for b, band in enumerate(save_bands):
            band_idx = band - subset_by_index[0]
            out[b]["vectors"][i] = eigenstates["vectors"][band_idx]
            out[b]["eigenvalues"][i] = eigenstates["eigenvalues"][band_idx]
    return out
