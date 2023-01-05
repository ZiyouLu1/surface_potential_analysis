from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from .energy_data.energy_eigenstates import (
    Eigenstate,
    EnergyEigenstates,
    get_eigenstate_list,
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
    de = eigenvalues[1:] - eigenvalues[0:-1]
    (line,) = a.plot(1 / moving_average(de))

    return fig, a, line


def plot_eigenvector_z(
    hamiltonian: SurfaceHamiltonian,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    z_points = np.linspace(hamiltonian.z_points[0], hamiltonian.z_points[-1], 1000)
    points = np.array(
        [(hamiltonian.delta_x / 2, hamiltonian.delta_y / 2, z) for z in z_points]
    )

    wfn = np.abs(hamiltonian.calculate_wavefunction_fast(points, eigenstate))
    (line,) = a.plot(z_points, wfn)

    return fig, a, line


def plot_eigenvector_through_bridge(
    hamiltonian: SurfaceHamiltonian,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    view: Literal["abs"] | Literal["angle"] = "abs",
) -> tuple[Figure, Axes, Line2D]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    x_points = np.linspace(hamiltonian.x_points[0], hamiltonian.x_points[-1], 1000)
    points = np.array([(x, hamiltonian.delta_y / 2, 0) for x in x_points])
    wfn = hamiltonian.calculate_wavefunction_fast(points, eigenstate)
    (line,) = ax1.plot(
        x_points - hamiltonian.delta_x / 2,
        np.abs(wfn) if view == "abs" else np.angle(wfn),
    )

    return fig, ax1, line


def plot_nth_eigenvector(
    hamiltonian: SurfaceHamiltonian, n=0, ax: Axes | None = None
) -> tuple[Figure, Axes]:

    e_vals, e_vecs = calculate_eigenvalues(hamiltonian, 0, 0)

    eigenvalue_index = np.argpartition(e_vals, n)[n]
    eigenvector = e_vecs[eigenvalue_index]

    fig, a = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    _, _, line = plot_eigenvector_z(hamiltonian, eigenvector, ax)
    line.set_label("Z direction")

    _, _, line = plot_eigenvector_through_bridge(hamiltonian, eigenvector, ax)
    line.set_label("X-Y through bridge")

    x_points = np.linspace(hamiltonian.x_points[0], hamiltonian.x_points[-1], 1000)
    points = np.array([(x, x, 0) for x in x_points])
    a.plot(
        np.sqrt(2) * (x_points - hamiltonian.delta_x / 2),
        np.abs(hamiltonian.calculate_wavefunction_fast(points, eigenvector)),
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


def plot_wavefunction_difference_in_xy(
    hamiltonian: SurfaceHamiltonian,
    eigenstate1: Eigenstate,
    eigenstate2: Eigenstate,
    ax: Axes | None = None,
    y_point=0.0,
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    x_points = np.linspace(0, hamiltonian.delta_x, 30)
    y_points = np.linspace(0, hamiltonian.delta_y, 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), y_point * np.ones_like(xv.ravel())]).T

    wfn1 = hamiltonian.calculate_wavefunction_fast(points, eigenstate1).reshape(
        xv.shape
    )
    wfn2 = hamiltonian.calculate_wavefunction_fast(points, eigenstate2).reshape(
        xv.shape
    )
    X = np.abs(wfn1) - np.abs(wfn2)

    im = ax1.imshow(np.abs(X))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)


def plot_wavefunction_in_xy(
    hamiltonian: SurfaceHamiltonian,
    eigenstate: Eigenstate,
    ax: Axes | None = None,
    y_point=0.0,
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    x_points = np.linspace(0, hamiltonian.delta_x, 30)
    y_points = np.linspace(0, hamiltonian.delta_y, 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), y_point * np.ones_like(xv.ravel())]).T

    X = hamiltonian.calculate_wavefunction_fast(points, eigenstate).reshape(xv.shape)
    im = ax1.imshow(np.abs(X))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)


def plot_wavepacket_in_xy(
    hamiltonian: SurfaceHamiltonian,
    eigenstates: EnergyEigenstates,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, AxesImage]:
    fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()

    x_points = np.linspace(-hamiltonian.delta_x, hamiltonian.delta_x, 60)
    y_points = np.linspace(0, hamiltonian.delta_y, 30)

    xv, yv = np.meshgrid(x_points, y_points)
    points = np.array([xv.ravel(), yv.ravel(), np.zeros_like(xv.ravel())]).T

    X = np.zeros_like(xv, dtype=complex)
    for eigenstate in get_eigenstate_list(eigenstates):
        print("i")
        wfn = hamiltonian.calculate_wavefunction_fast(points, eigenstate)
        X += (wfn).reshape(xv.shape)
    im = ax1.imshow(np.abs(X / len(eigenstates["eigenvectors"])))
    im.set_extent((x_points[0], x_points[-1], y_points[0], y_points[-1]))
    return (fig, ax1, im)
