from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalMomentumBasis
from surface_potential_analysis.basis_config.util import BasisConfigUtil

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
    )
    from surface_potential_analysis.eigenstate.eigenstate import EigenstateWithBasis

    from .wavepacket import WavepacketWithBasis

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _BX0Inv = TypeVar("_BX0Inv", bound=BasisLike[Any, Any])
    _BX1Inv = TypeVar("_BX1Inv", bound=BasisLike[Any, Any])
    _BX2Inv = TypeVar("_BX2Inv", bound=BasisLike[Any, Any])

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)


def get_furled_basis(
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    samples: tuple[_NS0Inv, _NS1Inv],
) -> BasisConfig[FundamentalMomentumBasis[int], FundamentalMomentumBasis[int], _BX2Inv]:
    """
    Given an basis and a sample size get the basis of the furled wavepacket.

    Parameters
    ----------
    basis : BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]
    samples : tuple[_NS0Inv, _NS1Inv]

    Returns
    -------
    BasisConfig[MomentumBasis[int], MomentumBasis[int], _BX2Inv]
    """
    (ns0, ns1) = samples
    return (
        FundamentalMomentumBasis(basis[0].delta_x / ns0, basis[0].n // ns0),
        FundamentalMomentumBasis(basis[1].delta_x / ns1, basis[1].n // ns1),
        basis[2],
    )


def furl_eigenstate(
    eigenstate: EigenstateWithBasis[
        FundamentalMomentumBasis[_L0Inv], FundamentalMomentumBasis[_L1Inv], _BX2Inv
    ],
    samples: tuple[_NS0Inv, _NS1Inv],
) -> WavepacketWithBasis[
    _NS0Inv,
    _NS1Inv,
    FundamentalMomentumBasis[int],
    FundamentalMomentumBasis[int],
    _BX2Inv,
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
        "basis": get_furled_basis(eigenstate["basis"], samples),
        "vectors": flattened * np.sqrt(ns0 * ns1),
        "energies": np.zeros(flattened.shape[0:2]),
    }


def get_unfurled_basis(
    basis: BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv],
    samples: tuple[_NS0Inv, _NS1Inv],
) -> BasisConfig[FundamentalMomentumBasis[int], FundamentalMomentumBasis[int], _BX2Inv]:
    """
    Given an basis and a sample size get the basis of the unfurled eigenstate.

    Parameters
    ----------
    basis : BasisConfig[_BX0Inv, _BX1Inv, _BX2Inv]
    samples : tuple[_NS0Inv, _NS1Inv]

    Returns
    -------
    BasisConfig[MomentumBasis[int], MomentumBasis[int], _BX2Inv]
    """
    (ns0, ns1) = samples
    return (
        FundamentalMomentumBasis(basis[0].delta_x / ns0, basis[0].n // ns0),
        FundamentalMomentumBasis(basis[1].delta_x * ns1, basis[1].n * ns1),
        basis[2],
    )


def get_wavepacket_unfurled_basis(
    wavepacket: WavepacketWithBasis[
        _NS0Inv,
        _NS1Inv,
        FundamentalMomentumBasis[_L0Inv],
        FundamentalMomentumBasis[_L1Inv],
        _BX2Inv,
    ]
) -> BasisConfig[FundamentalMomentumBasis[int], FundamentalMomentumBasis[int], _BX2Inv]:
    """
    Given a wavepacket get the basis of the unfurled eigenstate.

    Parameters
    ----------
    wavepacket : WavepacketWithBasis[ _NS0Inv, _NS1Inv, MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX2Inv ]

    Returns
    -------
    BasisConfig[MomentumBasis[int], MomentumBasis[int], _BX2Inv]
    """
    return get_unfurled_basis(wavepacket["basis"], wavepacket["energies"].shape)


def unfurl_wavepacket(
    wavepacket: WavepacketWithBasis[
        _NS0Inv,
        _NS1Inv,
        FundamentalMomentumBasis[_L0Inv],
        FundamentalMomentumBasis[_L1Inv],
        _BX2Inv,
    ]
) -> EigenstateWithBasis[
    FundamentalMomentumBasis[int], FundamentalMomentumBasis[int], _BX2Inv
]:
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
    (ns0, ns1) = wavepacket["energies"].shape
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
        "basis": get_wavepacket_unfurled_basis(wavepacket),
        "vector": flattened / np.sqrt(ns0 * ns1),
    }
