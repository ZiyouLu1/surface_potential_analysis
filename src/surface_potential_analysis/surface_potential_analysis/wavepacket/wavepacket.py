from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.axis import AxisWithLengthLike3d
from surface_potential_analysis.axis.axis import FundamentalPositionAxis
from surface_potential_analysis.basis.basis import (
    AxisWithLengthBasis,
    Basis3d,
    FundamentalMomentumBasis3d,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.basis.brillouin_zone import decrement_brillouin_zone
from surface_potential_analysis.basis.build import position_basis_from_shape
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateColllection3d,
    )

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)
_ND0Inv = TypeVar("_ND0Inv", bound=int)

_B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)

_B3d0Inv = TypeVar(
    "_B3d0Inv",
    bound=Basis3d[
        AxisWithLengthLike3d[Any, Any],
        AxisWithLengthLike3d[Any, Any],
        AxisWithLengthLike3d[Any, Any],
    ],
)

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


class Wavepacket(TypedDict, Generic[_S0Inv, _B0Inv]):
    """represents an approximation of a Wannier function."""

    basis: _B0Inv
    shape: _S0Inv
    # Vectors such that vectors.reshape(*shape, -1)[is0,is1, ..., :]
    # gives the eigenstate for the sample is0, is1, ...
    vectors: np.ndarray[tuple[int, int], np.dtype[np.complex_]]
    # Energies such that energies.reshape(*shape)[is0,is1, ..., isn]
    # gives the energies for the sample is0, is1, ...
    eigenvalues: np.ndarray[tuple[int], np.dtype[np.float_]]


_S03dInv = TypeVar("_S03dInv", bound=tuple[int, int, int])


Wavepacket3d = Wavepacket[_S03dInv, _B3d0Inv]
"""represents an approximation of a Wannier function."""


_A3d0Inv = TypeVar("_A3d0Inv", bound=AxisWithLengthLike3d[Any, Any])
_A3d1Inv = TypeVar("_A3d1Inv", bound=AxisWithLengthLike3d[Any, Any])
_A3d2Inv = TypeVar("_A3d2Inv", bound=AxisWithLengthLike3d[Any, Any])

Wavepacket3dWith2dSamples = Wavepacket3d[
    tuple[_NS0Inv, _NS1Inv, Literal[1]],
    _B3d0Inv,
]

WavepacketWithBasis3d = Wavepacket3d[
    tuple[_NS0Inv, _NS1Inv, Literal[1]],
    Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
]


PositionBasisWavepacket3d = Wavepacket3dWith2dSamples[
    _NS0Inv,
    _NS1Inv,
    FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv],
]

MomentumBasisWavepacket3d = Wavepacket3dWith2dSamples[
    _NS0Inv,
    _NS1Inv,
    FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv],
]


def save_wavepacket(path: Path, wavepacket: Wavepacket[Any, Any]) -> None:
    """
    Save a wavepacket in the npy format.

    Parameters
    ----------
    path : Path
    wavepacket :  Wavepacket[Any, Any]
    """
    np.save(path, wavepacket)


def load_wavepacket(
    path: Path,
) -> Wavepacket[Any, Any]:
    """
    Load a wavepacket from the npy format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Wavepacket[Any, Any, Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type: ignore[no-any-return]


def get_sample_basis(
    basis: AxisWithLengthBasis[_ND0Inv], shape: _S0Inv
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
        FundamentalPositionAxis(ax.delta_x * n, n)
        for (ax, n) in zip(basis, shape, strict=True)
    )


def get_unfurled_basis(
    basis: AxisWithLengthBasis[_ND0Inv], shape: _S0Inv
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
        FundamentalPositionAxis(ax.delta_x * n, ax.n * n)
        for (ax, n) in zip(basis, shape, strict=True)
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
    shape: _S0Inv,
) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
    """
    Get the frequencies of the samples in a wavepacket, as a fraction of dk.

    Parameters
    ----------
    shape : np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
    """
    sample_basis = position_basis_from_shape(shape)
    util = AxisWithLengthBasisUtil(sample_basis)
    return util.fundamental_nk_points / np.array(util.shape)[:, np.newaxis]  # type: ignore[no-any-return]


def get_wavepacket_sample_frequencies(
    basis: AxisWithLengthBasis[_ND0Inv], shape: _S0Inv
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
    sample_basis = get_sample_basis(basis, shape)
    util = AxisWithLengthBasisUtil(sample_basis)
    return util.fundamental_k_points  # type: ignore[return-value]


def get_wavepacket_sample_frequencies_first_brillouin_zone(
    basis: AxisWithLengthBasis[_ND0Inv], shape: _S0Inv
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
    sample_basis = get_sample_basis(basis, shape)
    util = AxisWithLengthBasisUtil(sample_basis)
    nk_points = util.nk_points
    for _ in util.shape:
        nk_points = decrement_brillouin_zone(basis, nk_points)
    return util.get_k_points_at_index(nk_points)  # type: ignore[return-value]


def as_eigenstate_collection(
    wavepacket: Wavepacket3dWith2dSamples[_NS0Inv, _NS1Inv, _B3d0Inv]
) -> EigenstateColllection3d[_B3d0Inv, int]:
    """
    Convert a wavepacket into an eigenstate collection.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _B3d0Inv]

    Returns
    -------
    EigenstateColllection[_B3d0Inv]
    """
    return {
        "basis": wavepacket["basis"],
        "bloch_fractions": get_wavepacket_sample_fractions(wavepacket["shape"]).reshape(
            3, -1
        ),
        "eigenvalues": wavepacket["eigenvalues"].reshape(-1),
        "vectors": wavepacket["vectors"].reshape(wavepacket["eigenvalues"].size, -1),
    }


def _from_eigenstate_collection(
    collection: EigenstateColllection3d[_B3d0Inv, _L0Inv],
    shape: _S0Inv,
) -> Wavepacket[_S0Inv, _B3d0Inv]:
    return {
        "basis": collection["basis"],
        "shape": shape,
        "eigenvalues": collection["eigenvalues"].reshape(np.prod(shape)),
        "vectors": collection["vectors"].reshape(np.prod(shape), -1),
    }


def generate_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[_ND0Inv], np.dtype[np.float_]]],
        SingleBasisOperator[_B0Inv],
    ],
    shape: _S0Inv,
    *,
    save_bands: np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
) -> list[Wavepacket[_S0Inv, _B0Inv]]:
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
    bloch_fractions = get_wavepacket_sample_fractions(shape)
    h = hamiltonian_generator(bloch_fractions[:, 0])
    basis_size = AxisWithLengthBasisUtil(h["basis"]).size
    save_bands = np.array([0]) if save_bands is None else save_bands
    subset_by_index: tuple[int, int] = (np.min(save_bands), np.max(save_bands))

    n_samples = np.prod(shape)
    vectors = np.empty((n_samples, basis_size), dtype=np.complex128)
    energies = np.empty(n_samples, dtype=np.float_)
    out: list[Wavepacket[_S0Inv, _B0Inv]] = [
        {
            "basis": h["basis"],
            "vectors": vectors.copy(),
            "eigenvalues": energies.copy(),
            "shape": shape,
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
