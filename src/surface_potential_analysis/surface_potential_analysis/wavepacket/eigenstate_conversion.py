from typing import Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import Basis, MomentumBasis
from surface_potential_analysis.basis_config.basis_config import BasisConfigUtil
from surface_potential_analysis.eigenstate.eigenstate import EigenstateWithBasis
from surface_potential_analysis.wavepacket.wavepacket import WavepacketWithBasis

_NS0Inv = TypeVar("_NS0Inv", bound=int)
_NS1Inv = TypeVar("_NS1Inv", bound=int)


_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


def furl_eigenstate(
    eigenstate: EigenstateWithBasis[MomentumBasis[int], MomentumBasis[int], _BX2Inv],
    samples: tuple[_NS0Inv, _NS1Inv],
) -> WavepacketWithBasis[
    _NS0Inv, _NS1Inv, MomentumBasis[int], MomentumBasis[int], _BX2Inv
]:
    """
    Convert an eigenstate into a wavepacket of a smaller unit cell.

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
    (ns0, ns1) = samples
    (nx0_old, nx1_old, nx2) = BasisConfigUtil(eigenstate["basis"]).shape
    (nx0, nx1) = (nx0_old // ns0, nx1_old // ns1)

    # We do the opposite to unfurl wavepacket at each step
    stacked = eigenstate["vector"].reshape(nx0 * ns0, nx1 * ns1, nx2)
    shifted = np.fft.fftshift(stacked, (0, 1))
    double_stacked = shifted.reshape(nx0, ns0, nx1, ns1, nx2)
    swapped = double_stacked.swapaxes(2, 3).swapaxes(0, 1).swapaxes(1, 2)
    unshifted = np.fft.ifftshift(swapped, axes=(0, 1, 2, 3))
    flattened = unshifted.reshape(ns0, ns1, -1)

    return {
        "basis": (
            {
                "_type": "momentum",
                "delta_x": eigenstate["basis"][0]["delta_x"] / ns0,
                "n": eigenstate["basis"][0]["n"] // ns0,
            },
            {
                "_type": "momentum",
                "delta_x": eigenstate["basis"][1]["delta_x"] / ns1,
                "n": eigenstate["basis"][1]["n"] // ns1,
            },
            eigenstate["basis"][2],
        ),
        "vectors": flattened,
    }


def unfurl_wavepacket(
    wavepacket: WavepacketWithBasis[
        _NS0Inv, _NS1Inv, MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX2Inv
    ]
) -> EigenstateWithBasis[MomentumBasis[int], MomentumBasis[int], _BX2Inv]:
    """
    Convert a wavepacket into an eigenstate of the irreducible unit cell.

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
    (ns0, ns1, _) = wavepacket["vectors"].shape
    (nx0, nx1, nx2) = BasisConfigUtil(wavepacket["basis"]).shape
    stacked = wavepacket["vectors"].reshape(ns0, ns1, nx0, nx1, nx2)

    # Shift negative frequency componets to the start, so we can
    # add the frequencies when we unravel
    shifted = np.fft.fftshift(stacked, axes=(0, 1, 2, 3))
    # We now have nx0, ns0, nx1, ns1, nx2
    swapped = shifted.swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3)
    # Ravel the samples into the eigenstates, since they have fractional frequencies
    # when we ravel the x0 and x1 axis we retain the order of the frequencies
    ravelled = swapped.reshape(ns0 * nx0, ns1 * nx1, nx2)
    # Shift the 0 frequency back to the start and flatten
    unshifted = np.fft.ifftshift(ravelled, axes=(0, 1))
    flattened = unshifted.reshape(-1)

    return {
        "basis": (
            {
                "_type": "momentum",
                "delta_x": wavepacket["basis"][0]["delta_x"] * ns0,
                "n": wavepacket["basis"][0]["n"] * ns0,
            },
            {
                "_type": "momentum",
                "delta_x": wavepacket["basis"][1]["delta_x"] * ns1,
                "n": wavepacket["basis"][1]["n"] * ns1,
            },
            wavepacket["basis"][2],
        ),
        "vector": flattened,
    }
