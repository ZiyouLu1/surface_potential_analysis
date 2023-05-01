from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import Basis
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
    PositionBasisConfig,
)
from surface_potential_analysis.eigenstate.eigenstate_calculation import (
    calculate_eigenstates,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from surface_potential_analysis.eigenstate.eigenstate import (
        Eigenstate,
    )
    from surface_potential_analysis.eigenstate.eigenstate_collection import (
        EigenstateColllection,
    )
    from surface_potential_analysis.hamiltonian import Hamiltonian

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_NS0Cov = TypeVar("_NS0Cov", bound=int, covariant=True)
_NS1Cov = TypeVar("_NS1Cov", bound=int, covariant=True)

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


class Wavepacket(TypedDict, Generic[_NS0Cov, _NS1Cov, _BC0Cov]):
    """represents an approximation of a Wannier function."""

    basis: _BC0Cov
    vectors: np.ndarray[tuple[_NS0Cov, _NS1Cov, int], np.dtype[np.complex_]]
    energies: np.ndarray[tuple[_NS0Cov, _NS1Cov], np.dtype[np.float_]]


WavepacketWithBasis = Wavepacket[
    _NS0Cov,
    _NS1Cov,
    BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
]


PositionBasisWavepacket = Wavepacket[
    _NS0Cov,
    _NS1Cov,
    PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv],
]

MomentumBasisWavepacket = Wavepacket[
    _NS0Cov,
    _NS1Cov,
    MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv],
]


def save_wavepacket(path: Path, wavepacket: Wavepacket[Any, Any, Any]) -> None:
    """
    Save a wavepacket in the npy format.

    Parameters
    ----------
    path : Path
    wavepacket : Wavepacket[Any, Any, Any]
    """
    np.save(path, wavepacket)


def load_wavepacket(path: Path) -> Wavepacket[Any, Any, Any]:
    """
    Load a wavepacket from the npy format.

    Parameters
    ----------
    path : Path

    Returns
    -------
    Wavepacket[Any, Any, Any]
    """
    return np.load(path, allow_pickle=True)[()]  # type:ignore[no-any-return]


def _get_wavepacket_sample_index(
    shape: np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.int_]]:
    n_x0 = np.fft.ifftshift(
        np.arange((-shape.item(0) + 1) // 2, (shape.item(0) - 1) // 2)
    )
    n_x1 = np.fft.ifftshift(
        np.arange((-shape.item(1) + 1) // 2, (shape.item(1) - 1) // 2)
    )

    x0v, x1v = np.meshgrid(n_x0, n_x1)
    return np.array([x0v, x1v])  # type: ignore[no-any-return]


def get_wavepacket_sample_fractions(
    shape: np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    """
    Get the frequencies of the samples in a wavepacket, as a fraction of dk.

    Parameters
    ----------
    shape : np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
    """
    # TODO: what should be the convention here
    x0v, x1v = np.meshgrid(
        np.fft.fftfreq(shape.item(0), 1),
        np.fft.fftfreq(shape.item(1), 1),
        indexing="ij",
    )
    return np.array([x0v, x1v])  # type: ignore[no-any-return]


def get_wavepacket_sample_frequencies(
    basis: BasisConfig[Any, Any, Any],
    shape: np.ndarray[tuple[Literal[2]], np.dtype[np.int_]],
) -> np.ndarray[tuple[Literal[3], int, int], np.dtype[np.float_]]:
    """
    Get the frequencies used in a given wavepacket.

    Parameters
    ----------
    basis : BasisConfig[Any, Any, Any]
    shape : np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]

    Returns
    -------
    np.ndarray[tuple[Literal[3], int, int], np.dtype[np.float_]]
    """
    util = BasisConfigUtil(basis)
    fractions = get_wavepacket_sample_fractions(shape)
    return (  # type: ignore[no-any-return]
        util.dk0[:, np.newaxis, np.newaxis] * fractions[np.newaxis, 0]
        + util.dk1[:, np.newaxis, np.newaxis] * fractions[np.newaxis, 1]
    )


def as_eigenstate_collection(
    wavepacket: Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]
) -> EigenstateColllection[_BC0Inv]:
    """
    Convert a wavepacket into an eigenstate collection.

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]

    Returns
    -------
    EigenstateColllection[_BC0Inv]
    """
    return {
        "basis": wavepacket["basis"],
        "bloch_phases": get_wavepacket_sample_frequencies(
            wavepacket["basis"], wavepacket["energies"].shape
        ).reshape(3, -1),
        "energies": wavepacket["energies"].reshape(-1),
        "vectors": wavepacket["vectors"].reshape(wavepacket["energies"].size, -1),
    }


def _from_eigenstate_collection(
    collection: EigenstateColllection[_BC0Inv], shape: tuple[_NS0Inv, _NS1Inv]
) -> Wavepacket[Any, Any, _BC0Inv]:
    return {
        "basis": collection["basis"],
        "energies": collection["energies"].reshape(shape),
        "vectors": collection["vectors"].reshape(*shape, -1),
    }


def generate_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]],
        Hamiltonian[_BC0Inv],
    ],
    samples: tuple[_NS0Inv, _NS1Inv],
    *,
    save_bands: np.ndarray[tuple[int], np.dtype[np.int_]] | None = None,
) -> list[Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]]:
    """
    Generate a wavepacket with the given number of samples.

    Parameters
    ----------
    hamiltonian_generator : Callable[[np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]], Hamiltonian[_BC0Inv]]
    samples : tuple[_NS0Inv, _NS1Inv]
    save_bands : np.ndarray[tuple[int], np.dtype[np.int_]] | None, optional

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]]]
    """
    h = hamiltonian_generator(np.array([0, 0, 0]))
    basis_length = len(BasisConfigUtil(h["basis"]))
    save_bands = np.array([0]) if save_bands is None else save_bands
    subset_by_index: tuple[int, int] = (np.min(save_bands), np.max(save_bands))

    vectors = np.empty((*samples, basis_length), dtype=np.complex128)
    energies = np.empty(samples, dtype=np.float_)
    out: list[Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]] = [
        {"basis": h["basis"], "vectors": vectors.copy(), "energies": energies.copy()}
        for _ in save_bands
    ]

    frequencies = get_wavepacket_sample_frequencies(h["basis"], np.array(samples))
    for i in range(np.prod(samples)):
        (is0, is1) = np.unravel_index(i, samples)
        h = hamiltonian_generator(frequencies[:, is0, is1])
        eigenstates = calculate_eigenstates(h, subset_by_index)

        for b, band in enumerate(save_bands):
            band_idx = band - subset_by_index[0]
            out[b]["vectors"][is0, is1] = eigenstates["vectors"][band_idx]
            out[b]["energies"][is0, is1] = eigenstates["energies"][band_idx]
    return out


def get_eigenstate(
    wavepacket: Wavepacket[
        _NS0Inv,
        _NS1Inv,
        _BC0Inv,
    ],
    idx: tuple[int, int] | int,
) -> Eigenstate[_BC0Inv]:
    """
    Get the eigenstate of a given wavepacket at a specific index.

    Parameters
    ----------
    wavepacket : Wavepacket[ _NS0Inv, _NS1Inv, _BC0Inv, ]
    idx : tuple[int, int]

    Returns
    -------
    Eigenstate[_BC0Inv]
    """
    unraveled_index = (
        idx
        if isinstance(idx, tuple)
        else (np.unravel_index(idx, wavepacket["energies"].shape))
    )
    return {
        "basis": wavepacket["basis"],
        "vector": wavepacket["vectors"][unraveled_index],
    }
