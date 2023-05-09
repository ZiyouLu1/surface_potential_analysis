def calculate_off_diagonal_energies(
    ft_potential: list[list[list[complex]]],
    resolution: tuple[float, float, float],
    dz: float,
    mass: float,
    sho_omega: float,
    z_offset: float,
) -> list[list[complex]]:
    """Calculate the off diagonal terms for the hamiltonian."""  # noqa: PYI021

def get_sho_wavefunction(
    z_points: list[float], sho_omega: float, mass: float, n: int
) -> list[float]:
    """Get the SHO wavefunction at the given z_points."""  # noqa: PYI021

def get_hermite_val(x: float, n: int) -> float:
    """Get the value of the nth hermite polynomial at x."""  # noqa: PYI021

def get_eigenstate_wavefunction(
    resolution: tuple[float, float, float],
    delta_x0: tuple[float, float],
    delta_x1: tuple[float, float],
    mass: float,
    sho_omega: float,
    kx: float,
    ky: float,
    vector: list[complex],
    points: list[tuple[float, float, float]],
) -> list[complex]:
    """Get the wavefunction for the given eigenstate."""  # noqa: PYI021
