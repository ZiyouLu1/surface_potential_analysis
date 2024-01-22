from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from surface_potential_analysis.basis.stacked_basis import StackedBasis
from surface_potential_analysis.dynamics.util import build_hop_operator
from surface_potential_analysis.util.interpolation import pad_ft_points

if TYPE_CHECKING:
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketWithEigenvalues,
    )


def build_hamiltonian_from_wavepackets(
    wavepackets: list[WavepacketWithEigenvalues[Any, Any]],
    basis: _B0Inv,
) -> SingleBasisOperator[_B0Inv]:
    (n_x1, n_x2, _) = basis.shape
    array = np.zeros((*basis.shape, *basis.shape), np.complex128)
    for i, wavepacket in enumerate(wavepackets):
        sample_shape = wavepacket["basis"][0].shape
        h = pad_ft_points(
            np.fft.fftn(
                wavepacket["eigenvalue"].reshape(sample_shape),
                norm="ortho",
            ),
            (3, 3, 1),
            (0, 1, 2),
        )
        for hop, hop_val in enumerate(h.ravel()):
            array[:, :, i, :, :, i] += hop_val * build_hop_operator(hop, (n_x1, n_x2))
    return {
        "data": array.reshape(-1),
        "basis": StackedBasis(basis, basis),
    }
