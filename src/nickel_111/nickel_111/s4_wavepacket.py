import numpy as np

from surface_potential_analysis.eigenstate.eigenstate import save_eigenstate
from surface_potential_analysis.wavepacket import (
    generate_wavepacket,
    load_wavepacket,
    normalize_wavepacket,
    save_wavepacket,
    unfurl_wavepacket,
)

from .s2_hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def generate_nickel_wavepacket():
    h = generate_hamiltonian(resolution=(23, 23, 12))
    save_bands = np.arange(20)

    wavepackets = generate_wavepacket(
        potential,
        samples=(8, 8),
        mass=1.6735575e-27,
        size=(4, 4),
        save_bands=save_bands,
    )
    for k, wavepacket in zip(save_bands, wavepackets):
        path = get_data_path(f"eigenstates_grid_{k}.json")
        save_wavepacket(path, wavepacket)


def generate_wavepacket_grid() -> None:
    for k in range(1):
        path = get_data_path(f"eigenstates_grid_{k}.json")
        wavepacket = load_wavepacket(path)
        wavepacket = normalize_wavepacket(wavepacket, (0, 0, 0))

        unfurled = unfurl_wavepacket(wavepacket)
        path = get_data_path(f"wavepacket_grid_{k}_traditional.json")
        save_eigenstate(unfurled, path)
