from typing import Literal

import numpy as np
from surface_potential_analysis.basis_config.basis_config import (
    MomentumBasisConfig,
)
from surface_potential_analysis.overlap.calculation import calculate_overlap
from surface_potential_analysis.overlap.overlap import save_overlap
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket,
    calculate_normalisation,
    convert_sho_wavepacket_to_momentum,
    normalize_momentum_wavepacket,
)

from nickel_111.s4_wavepacket import load_nickel_wavepacket

from .surface_data import get_data_path


def generate_fcc_wavepacket() -> (
    Wavepacket[
        Literal[8],
        Literal[8],
        MomentumBasisConfig[Literal[250], Literal[250], Literal[250]],
    ]
):
    wavepacket = load_nickel_wavepacket(0)
    momentum = convert_sho_wavepacket_to_momentum(wavepacket)
    return normalize_momentum_wavepacket(momentum, (0, 0, 117), 0)


def generate_next_fcc_wavepacket() -> (
    Wavepacket[
        Literal[8],
        Literal[8],
        MomentumBasisConfig[Literal[250], Literal[250], Literal[250]],
    ]
):
    """
    Generate a wavepacket grid of a neighboring fcc wavefunction.

    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket.

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next fcc site
    """
    wavepacket = load_nickel_wavepacket(0)
    momentum = convert_sho_wavepacket_to_momentum(wavepacket)
    return normalize_momentum_wavepacket(momentum, (0, 0, 117), 0)


def generate_hcp_wavepacket() -> (
    Wavepacket[
        Literal[8],
        Literal[8],
        MomentumBasisConfig[Literal[250], Literal[250], Literal[250]],
    ]
):
    wavepacket = load_nickel_wavepacket(1)
    momentum = convert_sho_wavepacket_to_momentum(wavepacket)
    return normalize_momentum_wavepacket(momentum, (0, 0, 117), 0)


def generate_next_hcp_wavepacket() -> (
    Wavepacket[
        Literal[8],
        Literal[8],
        MomentumBasisConfig[Literal[250], Literal[250], Literal[250]],
    ]
):
    """
    Generate a wavepacket grid of a neighboring hcp wavefunction.

    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket.

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next hcp site
    """
    wavepacket = load_nickel_wavepacket(1)
    momentum = convert_sho_wavepacket_to_momentum(wavepacket)
    return normalize_momentum_wavepacket(momentum, (0, 0, 117), 0)


def calculate_overlap_nickel() -> None:
    wavepacket_fcc = generate_fcc_wavepacket()
    # 0.9999999999996321
    print(calculate_normalisation(wavepacket_fcc))  # noqa: T201

    wavepacket_hcp = generate_hcp_wavepacket()
    # 0.9999999999997532
    print(calculate_normalisation(wavepacket_hcp))  # noqa: T201

    wavepacket_next_fcc = generate_next_fcc_wavepacket()
    wavepacket_next_hcp = generate_next_hcp_wavepacket()

    overlap_hcp_fcc = calculate_overlap(wavepacket_fcc, wavepacket_hcp)
    # -3.6208117396279577e-17 (should be 0)
    print(np.sum(overlap_hcp_fcc["vector"]))  # noqa: T201
    path = get_data_path("overlap_hcp_fcc.npy")
    save_overlap(path, overlap_hcp_fcc)

    overlap_fcc_fcc = calculate_overlap(wavepacket_fcc, wavepacket_next_fcc)
    # -1.381731140564679e-09 (should be 0)
    print(np.sum(overlap_fcc_fcc["vector"]))  # noqa: T201
    path = get_data_path("overlap_fcc_fcc.npy")
    save_overlap(path, overlap_fcc_fcc)

    overlap_hcp_hcp = calculate_overlap(wavepacket_hcp, wavepacket_next_hcp)
    # 4.12815207838777e-09
    print(np.sum(overlap_hcp_hcp["vector"]))  # noqa: T201
    path = get_data_path("overlap_hcp_hcp.npy")
    save_overlap(path, overlap_hcp_hcp)
