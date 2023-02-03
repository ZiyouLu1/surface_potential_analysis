from pathlib import Path
from typing import List, Tuple

import numpy as np

from surface_potential_analysis.energy_eigenstate import (
    EigenstateConfig,
    EigenstateConfigUtil,
    get_brillouin_points_irreducible_config,
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


def get_irreducible_config_nickel_111_supercell(
    config: EigenstateConfig,
) -> EigenstateConfig:
    return {
        "mass": config["mass"],
        "resolution": config["resolution"],
        "sho_omega": config["sho_omega"],
        "delta_x1": (config["delta_x1"][0], 0),
        "delta_x2": (config["delta_x1"][0] / 2, config["delta_x2"][1] / 2),
    }


def get_brillouin_points_nickel_111(
    config: EigenstateConfig, *, size: Tuple[int, int] = (8, 8), include_zero=True
):
    # Generate an equivalent config for the irreducible lattuice
    # Also note that delta_x2[1] = delta_x1[0] * sqrt(3)
    irreducible_config = get_irreducible_config_nickel_111_supercell(config)
    return get_brillouin_points_irreducible_config(
        irreducible_config, size=size, include_zero=include_zero
    )


def generate_energy_eigenstates_grid_nickel_111(
    path: Path,
    hamiltonian: SurfaceHamiltonianUtil,
    *,
    size: Tuple[int, int] = (8, 8),
    include_zero=True,
    include_bands: List[int] | None = None,
):
    k_points = get_brillouin_points_nickel_111(
        hamiltonian._config, size=size, include_zero=include_zero
    )

    return generate_energy_eigenstates_grid(
        path, hamiltonian, k_points, include_bands=include_bands
    )


def generate_eigenstates_grid():
    h = generate_hamiltonian(resolution=(12, 18, 12))
    path = get_data_path("eigenstates_grid_2.json")

    generate_energy_eigenstates_grid_nickel_111(
        path, h, size=(10, 6), include_bands=[0, 1, 2, 3]
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
    z_points = np.linspace(
        -util.characteristic_z * 2, util.characteristic_z * 2, 11
    )  # 21

    for (i, origin) in enumerate(origins):
        normalized = normalize_eigenstate_phase(eigenstates, origin)

        wavepacket = calculate_wavepacket_grid(normalized, x_points, y_points, z_points)
        path = get_data_path(f"eigenstates_wavepacket_{i}_small.json")
        save_wavepacket_grid_legacy(wavepacket, path)
