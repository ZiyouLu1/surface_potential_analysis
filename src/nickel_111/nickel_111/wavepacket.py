from pathlib import Path
from typing import List

import numpy as np

from surface_potential_analysis.brillouin_zone import (
    get_brillouin_points_irreducible_config,
)
from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    EigenstateConfigUtil,
    load_energy_eigenstates_old,
    normalize_eigenstate_phase,
)
from surface_potential_analysis.hamiltonian import (
    SurfaceHamiltonianUtil,
    generate_energy_eigenstates_grid,
)
from surface_potential_analysis.wavepacket_grid import (
    calculate_wavepacket_grid,
    save_wavepacket_grid_legacy,
)

from .hamiltonian import generate_hamiltonian
from .surface_data import get_data_path


def get_brillouin_points_nickel_111(
    config: EigenstateConfig, *, grid_size=4, include_zero=True
):
    # Generate an equivalent config for the irreducible lattuice
    # Also note that delta_x2[1] * sqrt(3) = delta_x1[0]
    irreducible_config: EigenstateConfig = {
        "mass": config["mass"],
        "resolution": config["resolution"],
        "sho_omega": config["sho_omega"],
        "delta_x1": (config["delta_x1"][0], 0),
        "delta_x2": (config["delta_x1"][0] / 2, config["delta_x2"][1] / 2),
    }
    return get_brillouin_points_irreducible_config(
        irreducible_config, grid_size=grid_size, include_zero=include_zero
    )


def generate_energy_eigenstates_grid_nickel_111(
    path: Path,
    hamiltonian: SurfaceHamiltonianUtil,
    *,
    grid_size=4,
    include_zero=True,
    include_bands: List[int] | None = None,
):
    k_points = get_brillouin_points_nickel_111(
        hamiltonian._config, grid_size=grid_size, include_zero=include_zero
    )

    return generate_energy_eigenstates_grid(
        path, hamiltonian, k_points, include_bands=include_bands
    )


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(12, 18, 12))
    path = get_data_path("eigenstates_grid_2.json")

    generate_energy_eigenstates_grid_nickel_111(
        path, h, grid_size=4, include_bands=[0, 1, 2, 3]
    )


def generate_wavepacket_grid():
    path = get_data_path("eigenstates_grid_2.json")
    eigenstates = load_energy_eigenstates_old(path)

    util = EigenstateConfigUtil(eigenstates["eigenstate_config"])

    origins = [
        (0, 1.0 * util.delta_x2[1] / 3, 0),
        (0, 2.0 * util.delta_x2[1] / 3, 0),
        # (util.delta_x / 2, 0.5 * util.delta_y / 3, 0),
        # (util.delta_x / 2, 2.5 * util.delta_y / 3, 0),
    ]

    x_points = np.linspace(-util.delta_x1[0] / 2, util.delta_x1[0] / 2, 13)  # 25
    y_points = np.linspace(0, util.delta_x2[1], 19)  # 37
    z_points = np.linspace(-util.delta_x1[0] / 2, util.delta_x1[0] / 2, 11)  # 21

    for (i, origin) in enumerate(origins):
        normalized = normalize_eigenstate_phase(eigenstates, origin)

        wavepacket = calculate_wavepacket_grid(normalized, x_points, y_points, z_points)
        path = get_data_path(f"eigenstates_wavepacket_{i}_small.json")
        save_wavepacket_grid_legacy(wavepacket, path)
