from surface_potential_analysis.hamiltonian import generate_energy_eigenstates_grid

from .hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(15, 15, 12))
    path = get_data_path("eigenstates_grid.json")

    generate_energy_eigenstates_grid(path, h, grid_size=4)
