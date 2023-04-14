from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis import Basis
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
    MomentumBasisConfig,
    PositionBasisConfig,
)
from surface_potential_analysis.eigenstate.eigenstate import Eigenstate
from surface_potential_analysis.eigenstate.eigenstate_calculation import (
    calculate_eigenstates,
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
    """
    An array of vectors for each state in the sample, spaced as np.fft.fftfreq(Ns, 1)
    """


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
    return np.load(path)[()]  # type:ignore[no-any-return]


def _get_wavepacket_sample_index(
    shape: np.ndarray[tuple[Literal[2], int, int], np.dtype[np.int_]]
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]:
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


def generate_wavepacket(
    hamiltonian_generator: Callable[
        [np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]],
        Hamiltonian[_BC0Inv],
    ],
    samples: tuple[_NS0Inv, _NS1Inv],
    *,
    save_bands: np.ndarray[tuple[int], np.dtype[np.int_]] | None = None
) -> np.ndarray[tuple[int], np.dtype[Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv]]]:
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

    vectors = np.empty((*samples, basis_length), dtype=np.complex128)
    out = np.array(
        [{"basis": h["basis"], "vectors": vectors} for _ in save_bands],
        dtype=Wavepacket[_NS0Inv, _NS1Inv, _BC0Inv],
    )

    frequencies = get_wavepacket_sample_frequencies(h["basis"], np.array(samples))
    for i, bloch_frequency in enumerate(frequencies.T.reshape(-1, 2)):
        h = hamiltonian_generator(bloch_frequency)
        eigenstates = calculate_eigenstates(h)

        (is0, is1) = np.unravel_index(i, samples)
        for b, band in enumerate(save_bands):
            out[b]["vectors"][is0, is1] = eigenstates["vectors"][band]
    return out  # type: ignore[no-any-return]


def get_global_phases(
    wavepacket: PositionBasisWavepacket[_NS0Inv, _NS1Inv, int, int, int],
    idx: int | tuple[int, int, int] = 0,
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    """
    Get the global bloch phase at a given index in the irreducible cell.

    Parameters
    ----------
    wavepacket : PositionBasisWavepacket[_NS0, _NS1, int, int, int]
        The wavepacket to get the global phase for
    idx : int | tuple[int, int, int], optional
        The index in ravelled or unravelled form, by default 0

    Returns
    -------
    np.ndarray[tuple[_NS0, _NS1], np.dtype[np.float_]]
        list of list of phases for each is0, is1 sampled in the wavepacket
    """
    util = BasisConfigUtil(wavepacket["basis"])
    stacked_idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    # Total phase given by k.x = dk * j/Nj * i * delta_x / Ni
    #                          = 2 * pi * j/Nj * i / Ni
    # j / Nj
    momentum_fractions = get_wavepacket_sample_fractions(
        np.array(wavepacket["vectors"].shape[0:2])
    )
    # i / Ni
    position_fractions = np.array([i / len(util) for i in stacked_idx[0:2]])[
        :, np.newaxis, np.newaxis
    ]

    return (  # type: ignore[no-any-return]
        2 * np.pi * np.sum(np.multiply(position_fractions, momentum_fractions), axis=0)
    )


W = TypeVar("W", bound=PositionBasisWavepacket[Any, Any, Any, Any, Any])


def normalize_wavepacket(
    wavepacket: W,
    idx: int | tuple[int, int, int] = 0,
    angle: float = 0,
) -> W:
    """
    Normalize the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[Any, Any, PositionBasis, PositionBasis, PositionBasis]
    idx : int | tuple[int, int, int], optional
        Index of the eigenstate to normalize, by default 0
        This index is taken in the irreducible unit cell
    angle: float, optional
        Angle to normalize the wavepacket to at the point idx.
        Each wavefunction will have the phase exp(i * angle) at the position
        given by idx

    Returns
    -------
    Wavepacket: Wavepacket[Any, Any, PositionBasis, PositionBasis, PositionBasis]
        The wavepacket, normalized
    """
    eigenvectors = wavepacket["vectors"]
    # TODO use irreducible basis here!!
    util = BasisConfigUtil(wavepacket["basis"])

    flat_idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    bloch_angles = np.angle(eigenvectors[:, :, flat_idx])
    global_phases: np.ndarray[tuple[int, int], np.dtype[np.float_]] = get_global_phases(
        wavepacket, idx
    )

    phases = np.exp(-1j * (bloch_angles + global_phases - angle))
    fixed_eigenvectors = eigenvectors * phases[:, :, np.newaxis]

    return {  # type: ignore[return-value]
        "basis": wavepacket["basis"],
        "vectors": fixed_eigenvectors,
    }


def select_wavepacket_eigenstate(
    wavepacket: Wavepacket[int, int, _BC0Inv], idx: tuple[int, int]
) -> Eigenstate[_BC0Inv]:
    """
    Select a specific eigenstate from the wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[int, int, _BC0Inv]
    idx : tuple[int, int]

    Returns
    -------
    Eigenstate[_BC0Inv]
    """
    return {
        "basis": wavepacket["basis"],
        "vector": wavepacket["vectors"][idx[0], idx[1]],
    }


def calculate_normalisation(wavepacket: Wavepacket[int, int, Any]) -> float:
    """
    calculate the normalization of a wavepacket.

    This should always be 1

    Parameters
    ----------
    wavepacket : Wavepacket[Any]

    Returns
    -------
    float
    """
    return np.sum(np.conj(wavepacket["vectors"]) * wavepacket["vectors"])
