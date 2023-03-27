from pathlib import Path
from typing import Any, Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.eigenstate.eigenstate import Eigenstate

from .basis import Basis, MomentumBasis, PositionBasis
from .basis_config import BasisConfig, BasisConfigUtil
from .hamiltonian import hamiltonian_in_basis
from .hamiltonian_builder import total_surface_hamiltonian
from .hamiltonian_eigenstates import calculate_eigenstates
from .potential import Potential

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_NS0Cov = TypeVar("_NS0Cov", bound=int, covariant=True)
_NS1Cov = TypeVar("_NS1Cov", bound=int, covariant=True)

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


class Wavepacket(TypedDict, Generic[_NS0Cov, _NS1Cov, _BX0Cov, _BX1Cov, _BX2Cov]):
    """
    represents an approximation of a Wannier function
    """

    basis: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
    vectors: np.ndarray[tuple[_NS0Cov, _NS1Cov, int], np.dtype[np.complex_]]
    """
    An array of vectors for each state in the sample, spaced as np.fft.fftfreq(Ns, 1)
    """


PositionBasisWavepacket = Wavepacket[
    _NS0Cov,
    _NS1Cov,
    PositionBasis[_L0Inv],
    PositionBasis[_L1Inv],
    PositionBasis[_L2Inv],
]

MomentumBasisWavepacket = Wavepacket[
    _NS0Cov,
    _NS1Cov,
    MomentumBasis[_L0Inv],
    MomentumBasis[_L1Inv],
    MomentumBasis[_L2Inv],
]


def save_wavepacket(
    path: Path, wavepacket: Wavepacket[Any, Any, Any, Any, Any]
) -> None:
    state = np.array(wavepacket, dtype=Wavepacket)
    np.save(path, state)


def load_wavepacket(path: Path) -> Wavepacket[Any, Any, Any, Any, Any]:
    return np.load(path)[()]  # type:ignore


def generate_wavepacket(
    potential: Potential[_L0Inv, _L1Inv, _L2Inv],
    samples: tuple[_NS0Inv, _NS1Inv],
    mass: float,
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    *,
    save_bands: np.ndarray[tuple[int], np.dtype[np.int_]] | None = None
) -> np.ndarray[
    tuple[int], np.dtype[Wavepacket[_NS0Inv, _NS1Inv, _BX0Inv, _BX1Inv, _BX2Inv]]
]:
    basis_length = len(BasisConfigUtil(basis))
    save_bands = np.array([0]) if save_bands is None else save_bands

    out = np.array(
        [
            {
                "basis": basis,
                "vectors": np.empty((*samples, basis_length), dtype=np.complex128),
            }
            for _ in save_bands
        ],
        dtype=Wavepacket[_NS0Inv, _NS1Inv, _BX0Inv, _BX1Inv, _BX2Inv],
    )
    for is0, s0 in enumerate(np.fft.fftfreq(samples[0], 1)):
        for is1, s1 in enumerate(np.fft.fftfreq(samples[1], 1)):
            h = total_surface_hamiltonian(
                potential, mass=mass, bloch_phase=np.array([s0, s1, 0])
            )
            h_in_basis = hamiltonian_in_basis(h, basis)
            eigenstates = calculate_eigenstates(h_in_basis)
            for i, band in enumerate(save_bands):
                out[i]["vectors"][is0, is1] = eigenstates["states"][band]
    return out


def get_wavepacket_index(
    shape: np.ndarray[tuple[Literal[2], int, int], np.dtype[np.int_]]
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.int_]]:
    n_x0 = np.fft.ifftshift(
        np.arange((-shape.item(0) + 1) // 2, (shape.item(0) - 1) // 2)
    )
    n_x1 = np.fft.ifftshift(
        np.arange((-shape.item(1) + 1) // 2, (shape.item(1) - 1) // 2)
    )

    x0v, x1v = np.meshgrid(n_x0, n_x1)
    return np.array([x0v, x1v])  # type: ignore


def get_wavepacket_fractions(
    shape: np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    # TODO: what should be the convention here
    x0v, x1v = np.meshgrid(
        np.fft.fftfreq(shape.item(0), 1),
        np.fft.fftfreq(shape.item(1), 1),
        indexing="ij",
    )
    return np.array([x0v, x1v])  # type: ignore


def get_global_phases(
    wavepacket: PositionBasisWavepacket[_NS0Inv, _NS1Inv, int, int, int],
    idx: int | tuple[int, int, int] = 0,
) -> np.ndarray[tuple[_NS0Inv, _NS1Inv], np.dtype[np.float_]]:
    """
    Get the global bloch phase at a given index in the irreducible cell

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
    momentum_fractions = get_wavepacket_fractions(
        np.array(wavepacket["eigenvectors"].shape[0:2])
    )
    # i / Ni
    position_fractions = np.array([i / len(util) for i in stacked_idx[0:2]])[
        :, np.newaxis, np.newaxis
    ]

    return (  # type: ignore
        2 * np.pi * np.sum(np.multiply(position_fractions, momentum_fractions), axis=0)
    )


W = TypeVar("W", bound=PositionBasisWavepacket[Any, Any, Any, Any, Any])


def normalize_wavepacket(
    wavepacket: W,
    idx: int | tuple[int, int, int] = 0,
    angle: float = 0,
) -> W:
    """
    Normalize the eigenstates in a wavepacket

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

    eigenvectors = wavepacket["eigenvectors"]
    # TODO use irreducible basis here!!
    util = BasisConfigUtil(wavepacket["basis"])

    flat_idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx
    bloch_angles = np.angle(eigenvectors[:, :, flat_idx])
    global_phases: np.ndarray[tuple[int, int], np.dtype[np.float_]] = get_global_phases(
        wavepacket, idx
    )

    phases = np.exp(-1j * (bloch_angles + global_phases - angle))
    fixed_eigenvectors = eigenvectors * phases[:, :, np.newaxis]

    return {  # type: ignore
        "basis": wavepacket["basis"],
        "eigenvectors": fixed_eigenvectors,
    }


def furl_eigenstate(
    eigenstate: Eigenstate[MomentumBasis[int], MomentumBasis[int], _BX2Cov],
    shape: tuple[_NS0Cov, _NS1Cov],
) -> Wavepacket[_NS0Cov, _NS1Cov, MomentumBasis[int], MomentumBasis[int], _BX2Cov]:
    """
    Convert an eigenstate into a wavepacket of a smaller unit cell

    Parameters
    ----------
    eigenstate : Eigenstate[MomentumBasis[_L0], MomentumBasis[_L1], _BX2]
        The eigenstate of the larger unit cell.
    shape : tuple[_NS0, _NS1]
        The shape of samples in the wavepacket grid.
        Note _NS0 must be a factor of _L0

    Returns
    -------
    Wavepacket[_NS0, _NS1, MomentumBasis[_L0 // _NS0], MomentumBasis[_L1 // _NS1], _BX2]
        The wavepacket with a smaller unit cell
    """
    raise NotImplementedError()
    return {
        "basis": (
            {
                "_type": "momentum",
                "dk": eigenstate["basis"][0]["dk"],
                "n": eigenstate["basis"][0]["n"] // shape[0],
            },
            {
                "_type": "momentum",
                "dk": eigenstate["basis"][1]["dk"],
                "n": eigenstate["basis"][1]["n"] // shape[1],
            },
            eigenstate["basis"][2],
        ),
        "eigenvectors": np.array([]),
    }


def unfurl_wavepacket(
    wavepacket: Wavepacket[
        _NS0Inv, _NS1Inv, MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX2Cov
    ]
) -> Eigenstate[MomentumBasis[int], MomentumBasis[int], _BX2Cov]:
    """
    Convert a wavepacket into an eigenstate of the irreducible unit cell

    Parameters
    ----------
    wavepacket : Wavepacket[_NS0, _NS1, MomentumBasis[_L0], MomentumBasis[_L1], _BX2]
        The wavepacket to unfurl

    Returns
    -------
    Eigenstate[MomentumBasis[_NS0 * _L0], MomentumBasis[_NS1 * _L1], _BX2]
        The eigenstate of the larger unit cell. Note this eigenstate has a
        smaller dk (for each axis dk = dk_i / NS)
    """
    (ns0, ns1, _) = wavepacket["eigenvectors"].shape
    raise NotImplementedError()
    return {
        "basis": (
            {
                "_type": "momentum",
                "dk": wavepacket["basis"][0]["dk"] / ns1,
                "n": wavepacket["basis"][0]["n"] * ns0,
            },
            {
                "_type": "momentum",
                "dk": wavepacket["basis"][1]["dk"] / ns1,
                "n": wavepacket["basis"][1]["n"] * ns1,
            },
            wavepacket["basis"][2],
        ),
        "vector": np.array([]),
    }
