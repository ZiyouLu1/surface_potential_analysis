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
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator_list import (
    SingleBasisDiagonalOperatorList,
    as_operator_list,
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
from surface_potential_analysis.state_vector.util import (
    get_most_localized_free_state_vectors,
)
from surface_potential_analysis.wavepacket.localization import (
    Wannier90Options,
    get_localization_operator_wannier90,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    LocalizationOperator,
    get_localized_hamiltonian_from_eigenvalues,
    get_localized_wavepackets,
    get_wavepacket_hamiltonian,
)
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_1d_k,
    plot_wavepacket_1d_x,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_unfurled_basis,
    get_wavepacket_basis,
    wavepacket_list_into_iter,
)

from sodium_copper_111.s4_wavepacket import (
    get_all_wavepackets,
    get_all_wavepackets_2d,
    get_all_wavepackets_2d_lithium,
    get_all_wavepackets_flat,
    get_all_wavepackets_flat_2d,
    get_all_wavepackets_flat_lithium,
    get_all_wavepackets_lithium,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListWithEigenvaluesList,
    )


_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

_SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])
_SB1 = TypeVar("_SB1", bound=StackedBasisLike[*tuple[Any, ...]])

FRICTION_NA_CU = 0.20 * 10**12


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
    hamiltonian: SingleBasisDiagonalOperatorList[FundamentalBasis[int], _SB0],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hamiltonian["data"].reshape(hamiltonian["basis"][0].n)
    # Stacked in, not z
    hamiltonian_stacked = as_operator_list(hamiltonian)

    hamiltonian_stacked = hamiltonian_stacked["data"].reshape(
        hamiltonian_stacked["basis"][0].n,
        *hamiltonian_stacked["basis"][1][0].shape,
        *hamiltonian_stacked["basis"][1][0].shape,
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

    convert_operator_to_basis(hamiltonian, basis)
    for i in range(1, transformed.ndim):
        slice_ = [0 for _ in range(transformed.ndim)]
        slice_[0] = slice(None)
        slice_[i] = 1
        energy = transformed[tuple(slice_)]
    return np.abs(energy / hbar)


def calculate_ballistic_rate(
    hamiltonian: SingleBasisDiagonalOperatorList[_B2, _SB1],
    operator: LocalizationOperator[_SB1, _B1, _B2],
) -> float:
    localized_hamiltonian = get_localized_hamiltonian_from_eigenvalues(
        hamiltonian, operator
    )
    print(localized_hamiltonian)  # noqa: T201
    # just the k=0 state
    k0_hamiltonian = localized_hamiltonian["data"].reshape(
        localized_hamiltonian["basis"].shape
    )[0]
    stacked = k0_hamiltonian.reshape(localized_hamiltonian["basis"][1].shape)
    print(stacked)  # noqa: T201

    # We need to find some average coherent rate between states
    # Only the middle state is actually surrounded by n//2 states,
    # some much less
    # TODO: how do we therefore calculate the rate a la first order
    # perterbation theory ...
    for i in range(1, transformed.ndim):
        slice_ = [0 for _ in range(transformed.ndim)]
        slice_[0] = slice(None)
        slice_[i] = 1
        energy = transformed[tuple(slice_)]
    return np.abs(energy / hbar)


def calculate_average_energy_per_band(
    hamiltonian: SingleBasisDiagonalOperatorList[FundamentalBasis[int], _B0],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hamiltonian_stacked = hamiltonian["data"].reshape(
        hamiltonian["basis"][0].n, hamiltonian["basis"][1][0].n
    )
    return np.abs(np.average(hamiltonian_stacked, axis=1))


def get_all_state_coherent_rate(
    hamiltonian: SingleBasisDiagonalOperatorList[_B0, _SB1],
) -> float:
    """
    Get the coherent rate of tunnelling between the localized states generated using the full basis.

    These states are simply eigenstates of position.

    Parameters
    ----------
    hamiltonian : SingleBasisDiagonalOperatorList[_B0, _SB1]

    Returns
    -------
    float
    """
    get_unfurled_basis(basis)
    position_basis


def get_bloch_wavefunction_list_where(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB1, _SB0],
    cond: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> BlochWavefunctionListWithEigenvaluesList[FundamentalBasis[int], _SB1, _SB0]:
    n_bands = np.count_nonzero(cond)
    return {
        "data": wavepackets["data"].reshape(cond.size, -1)[cond, :].reshape(-1),
        "basis": StackedBasis(
            StackedBasis(FundamentalBasis(n_bands), wavepackets["basis"][0][1]),
            wavepackets["basis"][1],
        ),
        "eigenvalue": wavepackets["eigenvalue"]
        .reshape(cond.size, -1)[cond, :]
        .reshape(-1),
    }


def get_localization_operator_1d(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB1, _SB0],
) -> LocalizationOperator[
    _SB0,
    StackedBasisLike[*tuple[FundamentalBasis[int], ...]],
    _B0,
]:
    n_bands = wavepackets["basis"][0][0].n
    projections = get_most_localized_free_state_vectors(
        get_wavepacket_basis(wavepackets), (n_bands,)
    )
    return get_localization_operator_wannier90(
        wavepackets,
        options=Wannier90Options(
            projection=projections, ignore_axes=(1, 2), num_iter=1000000
        ),
    )


def get_localized_wavepacket_where(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB1, _SB0],
    cond: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> BlochWavefunctionListWithEigenvaluesList[FundamentalBasis[int], _SB1, _SB0]:
    wavepackets_where = get_bloch_wavefunction_list_where(wavepackets, cond)
    operator = get_localization_operator_1d(wavepackets_where)

    return get_localized_wavepackets(wavepackets_where, operator)


def plot_initial_coherent_wavepackets(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB1, _SB0],
    temperature: float,
) -> None:
    hamiltonian = get_wavepacket_hamiltonian(wavepackets)
    energies = calculate_average_energy_per_band(hamiltonian)
    condition = energies < 4 * (Boltzmann * temperature)
    condition[0] = 1

    localized = get_localized_wavepacket_where(wavepackets, condition)

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    for i, wavepacket in enumerate(wavepacket_list_into_iter(localized)):
        _, _, ln = plot_wavepacket_1d_x(wavepacket, ax=ax0)
        ln.set_label(f"n={i}")

        _, _, ln = plot_wavepacket_1d_k(wavepacket, ax=ax1, measure="abs")
        ln.set_label(f"n={i}")
    fig0.show()
    fig1.show()
    input()


def plot_initial_coherent_wavepackets_sodium() -> None:
    temperature = 150
    wavepackets = get_all_wavepackets((4,), (32,))
    plot_initial_coherent_wavepackets(wavepackets, temperature)
    input()


def plot_ballistic_temperature_rate_curve(
    wavepackets: BlochWavefunctionListWithEigenvaluesList[_B0, _SB1, _SB0],
    temperatures: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    temperatures = np.linspace(50, 150, 100) if temperatures is None else temperatures

    energies = calculate_average_energy_per_band(
        get_wavepacket_hamiltonian(wavepackets)
    )

    rates = list[float]()
    for temperature in temperatures:
        initial_threshold = energies > 2 * (Boltzmann * temperature)
        initial_threshold[0] = 1

        initial_wavepackets = get_bloch_wavefunction_list_where(
            wavepackets, initial_threshold
        )
        hamiltonian = get_wavepacket_hamiltonian(initial_wavepackets)
        operator = get_localization_operator_1d(initial_wavepackets)
        approximate_rate = calculate_ballistic_rate(hamiltonian, operator)

        incoherent_rates = FRICTION_NA_CU * np.exp(
            (energies - energies[0]) / (Boltzmann * temperature)
        )
        # When coherent rate is greater than incoherent rate
        final_threshold = approximate_rate > incoherent_rates
        final_threshold[0] = 1
        final_wavepackets = get_bloch_wavefunction_list_where(
            wavepackets, final_threshold
        )
        hamiltonian = get_wavepacket_hamiltonian(final_wavepackets)
        operator = get_localization_operator_1d(final_wavepackets)
        final_rate = calculate_ballistic_rate(hamiltonian, operator)

        # TODO: Maybe repeat this for a second time ...
        rates.append(final_rate)
    (line,) = ax.plot(temperatures, rates)
    ax.set_xlabel("Temperature / k")
    ax.set_ylabel("Rate s-1")
    return fig, ax, line


def print_localized_hamiltonian() -> None:
    temperature = 150
    wavepackets = get_all_wavepackets((4,), (32,))
    energies = calculate_average_energy_per_band(
        get_wavepacket_hamiltonian(wavepackets)
    )
    initial_threshold = energies < 2 * (Boltzmann * temperature)
    initial_threshold[0] = 1

    initial_wavepackets = get_bloch_wavefunction_list_where(
        wavepackets, initial_threshold
    )
    hamiltonian = get_wavepacket_hamiltonian(initial_wavepackets)
    operator = get_localization_operator_1d(initial_wavepackets)
    print(calculate_ballistic_rate(hamiltonian, operator))  # noqa: T201


def plot_ballistic_rate_against_temperature_sodium() -> None:
    temperatures = np.linspace(0.1, 2000, 1000)

    wavepackets = get_all_wavepackets((16,), (32,))
    fig, ax, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures)
    line.set_label("1d 32")
    wavepackets = get_all_wavepackets((16,), (128,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("1d 128")
    wavepackets = get_all_wavepackets((16,), (256,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("1d 256")

    wavepackets = get_all_wavepackets_lithium((16,), (128,))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("1d 128")

    wavepackets = get_all_wavepackets_2d((8, 8), (32, 32))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("2d 32")
    wavepackets = get_all_wavepackets_2d((8, 8), (42, 42))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("2d 64")

    wavepackets = get_all_wavepackets_2d_lithium((8, 8), (32, 32))
    _, _, line = plot_ballistic_temperature_rate_curve(wavepackets, temperatures, ax=ax)
    line.set_label("2d 32")

    ax.set_title("Plot of ballistic rate against temperature for 1d model of Na on Cu")
    ax.legend()
    ax.set_xlim(0, temperatures[-1])
    ax.set_ylim(0, None)
    fig.show()
    input()


def plot_ballistic_rate_against_against_temperature_flat() -> None:
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


def plot_ballistic_rate_against_temperature_flat_large_temperature() -> None:
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

    ax.set_title(
        "Plot of ballistic rate against temperature. Flat surface, large temperature"
    )
    ax.legend()
    ax.set_xlim(0, temperatures[-1])
    ax.set_ylim(0, None)
    fig.show()
    input()
