import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .eigenstate_plot import plot_eigenstate
from .hamiltonian import SurfaceHamiltonianUtil


def plot_energy_eigenvalues(
    hamiltonian: SurfaceHamiltonianUtil, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues, _ = hamiltonian.calculate_eigenvalues(0, 0)
    for e in eigenvalues:
        a.plot([0, 1], [e, e])

    return fig, a


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_density_of_states(
    hamiltonian: SurfaceHamiltonianUtil, ax: Axes | None = None
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues, _ = hamiltonian.calculate_eigenvalues(0, 0)
    de = np.subtract(eigenvalues[1:], eigenvalues[0:-1])
    (line,) = a.plot(1 / moving_average(de))

    return fig, a, line


def plot_nth_eigenstate(
    hamiltonian: SurfaceHamiltonianUtil, n=0, ax: Axes | None = None
):
    e_vals, e_vec = hamiltonian.calculate_eigenvalues(
        hamiltonian.dkx1[0] / 2, hamiltonian.dkx2[1] / 2
    )

    eigenvalue_index = np.argpartition(e_vals, n)[n]
    eigenstate = e_vec[eigenvalue_index]
    fig, a = plot_eigenstate(hamiltonian._config, eigenstate, ax)
    a.set_title(f"Plot of the n={n} wavefunction")
    return fig, a


def plot_first_4_eigenvectors(hamiltonian: SurfaceHamiltonianUtil) -> Figure:
    fig, axs = plt.subplots(2, 2)
    axes = [axs[0][0], axs[1][0], axs[0][1], axs[1][1]]

    e_vals, e_vec = hamiltonian.calculate_eigenvalues(
        hamiltonian.dkx1[0] / 2, hamiltonian.dkx2[1] / 2
    )

    for (n, ax) in enumerate(axes):
        eigenvalue_index = np.argpartition(e_vals, n)[n]
        eigenstate = e_vec[eigenvalue_index]
        plot_eigenstate(hamiltonian._config, eigenstate, ax=ax)
        ax.set_title(f"Plot of the n={n} wavefunction")

    fig.tight_layout()

    return fig


def plot_bands_occupation(
    hamiltonian: SurfaceHamiltonianUtil,
    temperature: float = 50.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    eigenvalues, _ = hamiltonian.calculate_eigenvalues(0, 0)
    normalized_eigenvalues = eigenvalues - np.min(eigenvalues)
    beta = 1 / (scipy.constants.Boltzmann * temperature)
    occupations = np.exp(-normalized_eigenvalues * beta)
    (line,) = a.plot(occupations / np.sum(occupations))

    return fig, a, line
