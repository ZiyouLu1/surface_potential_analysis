from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis.conversion import (
    basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.basis.util import Basis3dUtil, BasisUtil
from surface_potential_analysis.wavepacket.conversion import convert_wavepacket_to_basis
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    Wavepacket3d,
    get_unfurled_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import (
        FundamentalMomentumAxis,
        FundamentalMomentumAxis3d,
    )
    from surface_potential_analysis.axis.axis_like import AxisLike3d
    from surface_potential_analysis.basis.basis import Basis, Basis3d
    from surface_potential_analysis.eigenstate.eigenstate import (
        Eigenstate,
        Eigenstate3dWithBasis,
    )

    from .wavepacket import WavepacketWithBasis3d

    _NS0Inv = TypeVar("_NS0Inv", bound=int)
    _NS1Inv = TypeVar("_NS1Inv", bound=int)

    _A3d0Inv = TypeVar("_A3d0Inv", bound=AxisLike3d[Any, Any])
    _A3d1Inv = TypeVar("_A3d1Inv", bound=AxisLike3d[Any, Any])
    _A3d2Inv = TypeVar("_A3d2Inv", bound=AxisLike3d[Any, Any])

    _B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
    _BM0Inv = TypeVar("_BM0Inv", bound=tuple[FundamentalMomentumAxis[Any, Any], ...])
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])

    _S3d0Inv = TypeVar("_S3d0Inv", bound=tuple[int, int, int])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)


def furl_eigenstate(
    eigenstate: Eigenstate3dWithBasis[
        FundamentalMomentumAxis3d[_L0Inv], FundamentalMomentumAxis3d[_L1Inv], _A3d2Inv
    ],
    shape: tuple[_NS0Inv, _NS1Inv, Literal[1]],
) -> WavepacketWithBasis3d[
    _NS0Inv,
    _NS1Inv,
    FundamentalMomentumAxis3d[int],
    FundamentalMomentumAxis3d[int],
    _A3d2Inv,
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
    (ns0, ns1, _) = shape
    (nx0_old, nx1_old, nx2) = Basis3dUtil(eigenstate["basis"]).shape
    (nx0, nx1) = (nx0_old // ns0, nx1_old // ns1)

    # We do the opposite to unfurl wavepacket at each step
    stacked = eigenstate["vector"].reshape(nx0 * ns0, nx1 * ns1, nx2)
    shifted = np.fft.fftshift(stacked, (0, 1))
    double_stacked = shifted.reshape(nx0, ns0, nx1, ns1, nx2)
    swapped = double_stacked.swapaxes(2, 3).swapaxes(0, 1).swapaxes(1, 2)
    unshifted = np.fft.ifftshift(swapped, axes=(0, 1, 2, 3))
    flattened = unshifted.reshape(ns0, ns1, -1)

    basis = get_unfurled_basis(eigenstate["basis"], shape)
    return {
        "basis": basis_as_fundamental_momentum_basis(basis),  # type: ignore[typeddict-item]
        "shape": shape,
        "vectors": flattened * np.sqrt(ns0 * ns1),
        "energies": np.zeros(flattened.shape[0:2]),
    }


def _unfurl_momentum_basis_wavepacket(
    wavepacket: Wavepacket[_S0Inv, _BM0Inv]
) -> Eigenstate[tuple[FundamentalMomentumAxis[Any, Any], ...]]:
    sample_shape = wavepacket["shape"]
    states_shape = BasisUtil(wavepacket["basis"]).shape
    final_shape = tuple(
        ns * nx for (ns, nx) in zip(sample_shape, states_shape, strict=True)
    )
    stacked = wavepacket["vectors"].reshape(*sample_shape, *states_shape)

    # Shift negative frequency componets to the start, so we can
    # add the frequencies when we unravel
    shifted = np.fft.fftshift(stacked)
    # The list of axis index n,0,n+1,1,...,2n-1,n-1
    nd = len(sample_shape)
    locations = [
        x for y in zip(range(nd, 2 * nd), range(0, nd), strict=True) for x in y
    ]
    # We now have nx0, ns0, nx1, ns1, ns2, ...
    swapped = np.transpose(shifted, axes=locations)
    # Ravel the samples into the eigenstates, since they have fractional frequencies
    # when we ravel the x0 and x1 axis we retain the order of the frequencies
    ravelled = swapped.reshape(*final_shape)
    # Shift the 0 frequency back to the start and flatten
    unshifted = np.fft.ifftshift(ravelled)
    flattened = unshifted.reshape(-1)

    basis = get_unfurled_basis(wavepacket["basis"], wavepacket["shape"])
    return {
        "basis": basis_as_fundamental_momentum_basis(basis),
        "vector": flattened / np.sqrt(np.prod(sample_shape)),
    }


@overload
def unfurl_wavepacket(
    wavepacket: Wavepacket3d[_S3d0Inv, _B3d0Inv]
) -> Eigenstate[
    tuple[
        FundamentalMomentumAxis[Any, Literal[3]],
        FundamentalMomentumAxis[Any, Literal[3]],
        FundamentalMomentumAxis[Any, Literal[3]],
    ]
]:
    ...


@overload
def unfurl_wavepacket(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> (
    Eigenstate[tuple[FundamentalMomentumAxis[Any, Any], ...]]
    | Eigenstate[
        tuple[
            FundamentalMomentumAxis[Any, Literal[3]],
            FundamentalMomentumAxis[Any, Literal[3]],
            FundamentalMomentumAxis[Any, Literal[3]],
        ]
    ]
):
    ...


def unfurl_wavepacket(wavepacket: Wavepacket[_S0Inv, _B0Inv]) -> Eigenstate[Any]:
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
    converted_basis = basis_as_fundamental_momentum_basis(wavepacket["basis"])
    converted = convert_wavepacket_to_basis(wavepacket, converted_basis)
    return _unfurl_momentum_basis_wavepacket(converted)
