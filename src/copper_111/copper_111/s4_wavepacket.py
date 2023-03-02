from surface_potential_analysis.hamiltonian import generate_energy_eigenstates_grid

from .s2_hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(23, 23, 12))
    save_bands = {k: get_data_path(f"eigenstates_grid_{k}.json") for k in range(20)}

    generate_energy_eigenstates_grid(h, size=(4, 4), save_bands=save_bands)
