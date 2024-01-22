from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann, hbar
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalTransformedPositionBasis,
)
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_diagonal_schrodinger_equation,
)
from surface_potential_analysis.probability_vector.plot import plot_probability_1d_x
from surface_potential_analysis.probability_vector.probability_vector import (
    from_state_vector_list,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_wavepacket_hamiltonian,
)

from sodium_copper_111.s4_wavepacket import (
    get_all_wavepackets,
    get_all_wavepackets_2d,
    get_all_wavepackets_flat,
    get_all_wavepackets_flat_2d,
    get_all_wavepackets_flat_lithium,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketWithEigenvaluesList,
    )


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])

_SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
_SB1 = TypeVar("_SB1", bound=StackedBasisLike[*tuple[Any, ...]])


def aaa() -> None:
    hamiltonian = get_wavepacket_hamiltonian_sodium((16,), (32,))
    initial_state: StateVector[Any] = {
        "basis": hamiltonian["basis"][0],
        "data": np.zeros_like(hamiltonian["data"]),
    }
    initial_state["data"][np.arange(8)] = 1 / 8
    hamiltonian["data"] -= np.min(hamiltonian["data"])
    de = np.max(hamiltonian["data"][np.arange(8)]) - np.min(
        hamiltonian["data"][np.arange(8)]
    )
    times = EvenlySpacedTimeBasis(5, 1, 0, 3 * hbar / de)
    states = solve_diagonal_schrodinger_equation(initial_state, times, hamiltonian)
    basis = StackedBasis(
        states["basis"][0],
        StackedBasis(
            FundamentalTransformedPositionBasis(
                np.array([1]) * initial_state["basis"][1].n, initial_state["basis"].n
            )
        ),
    )
    probabilities = from_state_vector_list(
        convert_state_vector_list_to_basis(
            {"basis": basis, "data": states["data"]},
            stacked_basis_as_fundamental_position_basis(basis[1]),
        )
    )
    fig, ax = plt.subplots()
    for idx, _t in enumerate(times.times):
        basis = StackedBasis(
            FundamentalTransformedPositionBasis(
                np.array([1]) * initial_state["basis"][1].n, initial_state["basis"].n
            )
        )

        plot_probability_1d_x(
            {
                "basis": probabilities["basis"][1],
                "data": probabilities["data"].reshape(probabilities["basis"].shape)[
                    idx
                ],
            },
            ax=ax,
        )
    fig.show()
    input()


def calculate_ballistic_rate_per_band(
    hamiltonian: SingleBasisDiagonalOperator[
        StackedBasisLike[FundamentalBasis[int], _SB0]
    ]
) -> np.ndarray[Any, np.dtype[np.float_]]:
    diagonal_stacked_hamiltonian = hamiltonian["data"].reshape(
        hamiltonian["basis"][0].shape
    )
    hamiltonian_stacked = np.zeros(
        (*hamiltonian["basis"][0].shape, hamiltonian["basis"][0][1].n),
        dtype=np.complex_,
    )
    for band in range(hamiltonian["basis"][0][0].n):
        hamiltonian_stacked[band] = np.diag(diagonal_stacked_hamiltonian[band])
    hamiltonian_stacked = hamiltonian_stacked.reshape(
        hamiltonian["basis"][0][0].n,
        *hamiltonian["basis"][0][1].shape,
        *hamiltonian["basis"][0][1].shape,
    )

    transformed = np.fft.ifftn(
        np.fft.fftn(
            hamiltonian_stacked,
            axes=list(range(1, 1 + hamiltonian["basis"][0][1].ndim)),
            norm="ortho",
        ),
        axes=list(
            range(
                1 + hamiltonian["basis"][0][1].ndim,
                1 + 2 * hamiltonian["basis"][0][1].ndim,
            )
        ),
        norm="ortho",
    )
    for i in range(1, transformed.ndim):
        slice_ = [0 for _ in range(transformed.ndim)]
        slice_[0] = slice(None)
        slice_[i] = 1
        energy = transformed[tuple(slice_)]
    return np.abs(energy / hbar)


def calculate_average_energy_per_band(
    hamiltonian: SingleBasisDiagonalOperator[
        StackedBasisLike[FundamentalBasis[int], _B0]
    ]
) -> np.ndarray[Any, np.dtype[np.float_]]:
    hamiltonian_stacked = hamiltonian["data"].reshape(*hamiltonian["basis"][0].shape)
    return np.abs(np.average(hamiltonian_stacked, axis=1))


def plot_ballistic_temperature_rate_curve(
    wavepackets: WavepacketWithEigenvaluesList[_B0, _SB1, _SB0],
    temperatures: np.ndarray[tuple[int], np.dtype[np.float_]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    temperatures = np.linspace(50, 150, 100) if temperatures is None else temperatures

    hamiltonian = get_wavepacket_hamiltonian(wavepackets)
    energies = calculate_average_energy_per_band(hamiltonian)
    rates = calculate_ballistic_rate_per_band(hamiltonian)

    average_rates = list[float]()
    for temperature in temperatures:
        weights = np.exp(-energies / (Boltzmann * temperature))
        average_rate = np.average(rates, weights=weights)
        average_rates.append(average_rate)
    (line,) = ax.plot(temperatures, average_rates)
    ax.set_xlabel("Temperature / k")
    ax.set_ylabel("Rate s-1")
    return fig, ax, line


def plot_ballistic_rate_against_t_sodium() -> None:
    temperatures = np.linspace(0.1, 2000, 1000)

    wavepackets = get_all_wavepackets((16,), (32,))
    fig, ax, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures)
    line.set_label("1d 32")
    wavepackets = get_all_wavepackets((16,), (64,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("1d 64")
    wavepackets = get_all_wavepackets((16,), (256,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("1d 256")
    wavepackets = get_all_wavepackets((16,), (512,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("1d 512")

    wavepackets = get_all_wavepackets_2d((8, 8), (32, 32))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("2d 32")
    wavepackets = get_all_wavepackets_2d((8, 8), (42, 42))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("2d 64")

    ax.set_title("Plot of ballistic rate against temperature for 1d model of Na on Cu")
    ax.legend()
    ax.set_xlim(0, temperatures[-1])
    ax.set_ylim(0, None)
    fig.show()
    input()


def plot_ballistic_rate_against_t_flat() -> None:
    temperatures = np.linspace(0.01, 200, 10000)

    wavepackets = get_all_wavepackets_flat((128,), (32,))
    fig, ax, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures)
    line.set_label("Na. 128 samples, 32 bands")
    wavepackets = get_all_wavepackets_flat((16,), (32,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Na. 16 samples, 32 bands")
    wavepackets = get_all_wavepackets_flat((16,), (128,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Na. 16 samples, 128 bands")

    wavepackets = get_all_wavepackets_flat_lithium((128,), (32,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Li. 128 samples, 32 bands")

    wavepackets = get_all_wavepackets_flat_2d((8, 8), (32, 32))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Na 2d. 16 samples, 128 bands")

    ax.set_title("Plot of ballistic rate against temperature. Flat surface")
    ax.legend()
    ax.set_xlim(0, temperatures[-1])
    ax.set_ylim(0, None)
    fig.show()
    input()


def plot_ballistic_rate_against_mass() -> None:
    temperatures = np.linspace(0.01, 2000, 1000)

    wavepackets = get_all_wavepackets_flat((128,), (32,))
    fig, ax, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures)
    line.set_label("Na. 128 samples, 32 bands")
    wavepackets = get_all_wavepackets_flat((16,), (32,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Na. 16 samples, 32 bands")
    wavepackets = get_all_wavepackets_flat((16,), (128,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Na. 16 samples, 128 bands")

    wavepackets = get_all_wavepackets_flat_2d((8, 8), (32, 32))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("Na 2d. 16 samples, 128 bands")

    ax.set_title("Plot of ballistic rate against temperature. Flat surface")
    ax.legend()
    ax.set_xlim(0, temperatures[-1])
    ax.set_ylim(0, None)
    fig.show()
    input()
