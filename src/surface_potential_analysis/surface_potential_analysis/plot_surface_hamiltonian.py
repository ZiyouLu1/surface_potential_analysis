import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .energy_data.plot_eigenstate import (
    plot_eigenstate_through_bridge,
    plot_eigenstate_z,
)
from .hamiltonian import SurfaceHamiltonian, calculate_eigenvalues


def plot_energy_eigenvalues(
    hamiltonian: SurfaceHamiltonian, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues, _ = calculate_eigenvalues(hamiltonian, 0, 0)
    for e in eigenvalues:
        a.plot([0, 1], [e, e])

    return fig, a


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_density_of_states(
    hamiltonian: SurfaceHamiltonian, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues, _ = calculate_eigenvalues(hamiltonian, 0, 0)
    de = np.subtract(eigenvalues[1:], eigenvalues[0:-1])
    (line,) = a.plot(1 / moving_average(de))

    return fig, a, line


def plot_nth_eigenvector(
    hamiltonian: SurfaceHamiltonian, n=0, ax: Axes | None = None
) -> tuple[Figure, Axes]:

    e_vals, e_vecs = calculate_eigenvalues(
        hamiltonian, hamiltonian.dkx / 2, hamiltonian.dky / 2
    )

    eigenvalue_index = np.argpartition(e_vals, n)[n]
    eigenstate = e_vecs[eigenvalue_index]

    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, line = plot_eigenstate_z(hamiltonian._config, eigenstate, ax)
    line.set_label("Z direction")

    _, _, line = plot_eigenstate_through_bridge(hamiltonian._config, eigenstate, ax)
    line.set_label("X-Y through bridge")

    x_points = np.linspace(hamiltonian.x_points[0], hamiltonian.x_points[-1], 1000)
    points = np.array([(x, x, 0) for x in x_points])
    a.plot(
        np.sqrt(2) * (x_points - hamiltonian.delta_x / 2),
        np.abs(hamiltonian.calculate_wavefunction_fast(eigenstate, points)),
        label="X-Y through Top",
    )
    a.set_title(f"Plot of the n={n} wavefunction")
    a.legend()

    return (fig, a)


def plot_first_4_eigenvectors(hamiltonian: SurfaceHamiltonian) -> Figure:
    fig, axs = plt.subplots(2, 2)
    axes = [axs[0][0], axs[1][0], axs[0][1], axs[1][1]]
    for (n, ax) in enumerate(axes):
        plot_nth_eigenvector(hamiltonian, n, ax=ax)

    fig.tight_layout()

    return fig


def plot_bands_occupation(
    hamiltonian: SurfaceHamiltonian, temperature: float = 50.0, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues, _ = calculate_eigenvalues(hamiltonian, 0, 0)
    normalized_eigenvalues = eigenvalues - np.min(eigenvalues)
    beta = 1 / (scipy.constants.Boltzmann * temperature)
    occupations = np.exp(-normalized_eigenvalues * beta)
    (line,) = a.plot(occupations / np.sum(occupations))

    return fig, a, line