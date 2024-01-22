from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.time_basis_like import (
    EvenlySpacedTimeBasis,
    FundamentalTimeBasis,
)
from surface_potential_analysis.dynamics.isf import (
    ISFFey4VariableFit,
    calculate_isf_approximate_locations,
    fit_isf_to_fey_4_variable_model_110,
)
from surface_potential_analysis.dynamics.isf_plot import (
    plot_isf_against_time,
    plot_isf_fey_4_variable_fit_against_time,
)
from surface_potential_analysis.dynamics.plot import plot_average_probability_per_band
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation,
)
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    get_simplified_collapse_operators_from_a_matrix,
    solve_stochastic_schrodinger_equation,
)
from surface_potential_analysis.operator.operator import (
    StatisticalDiagonalOperator,
    average_eigenvalues,
)
from surface_potential_analysis.probability_vector.plot import (
    plot_probability_against_time,
)
from surface_potential_analysis.probability_vector.probability_vector import (
    ProbabilityVectorList,
    average_probabilities,
    from_state_vector,
    from_state_vector_list,
    get_probability,
)
from surface_potential_analysis.util.decorators import npy_cached_dict

from hydrogen_nickel_111.experimental_data import get_experiment_data
from hydrogen_nickel_111.s4_wavepacket import get_hydrogen_energy_difference
from hydrogen_nickel_111.s6_a_calculation import get_tunnelling_a_matrix_hydrogen
from hydrogen_nickel_111.s6_incoherent_isf import get_jianding_isf_112bar
from hydrogen_nickel_111.s6_schrodinger_dynamics import (
    get_coherent_hamiltonian,
    plot_expected_occupation_per_band_hydrogen,
)

from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

_B0 = TypeVar("_B0", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def _sse_sim_cache(
    temperature: float,
    idx: int,
    times: Any,
    _i: int = 0,  # noqa: ARG001,ANN401
) -> Path:
    return get_data_path(f"dynamics/see_simulation_{temperature}K_{idx}_{_i}")


@npy_cached_dict(_sse_sim_cache, load_pickle=True)
def get_simulation_at_temperature(
    temperature: float, idx: int, times: _B0, _i: int = 0
) -> StateVectorList[StackedBasisLike[FundamentalBasis[Literal[20]], _B0], Any]:
    a_matrix = get_tunnelling_a_matrix_hydrogen((10, 10), 8, temperature)
    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = get_coherent_hamiltonian(a_matrix["basis"][0])
    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"][0],
        "data": np.zeros(a_matrix["basis"][0].n, dtype=np.complex128),
    }
    initial_state["data"][idx] = 1
    return solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )


def get_repeated_simulation_at_temperature(
    temperature: float, idx: int, times: _B0, repeats: int = 1
) -> StateVectorList[StackedBasisLike[FundamentalBasis[int], _B0], Any]:
    states_i = get_simulation_at_temperature(temperature, idx, times, _i=0)
    probabilities = np.zeros((0, times.times.size, 800), dtype=np.complex128)
    for repeat in range(repeats):
        states_i = get_simulation_at_temperature(temperature, idx, times, _i=repeat)
        probabilities = np.append(
            probabilities,
            states_i["data"].reshape(*states_i["basis"][0].shape, -1),
            axis=0,
        )

    return {
        "basis": StackedBasis(
            StackedBasis(FundamentalBasis(repeats * 20), times), states_i["basis"][1]
        ),
        "data": probabilities.reshape(-1),
    }


def plot_average_isf_against_time() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((6, 6), 6, 150)
    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = get_coherent_hamiltonian(a_matrix["basis"][0])
    hamiltonian["data"] = np.zeros_like(hamiltonian["data"])

    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"][0],
        "data": np.zeros(a_matrix["basis"].shape[0], dtype=np.complex128),
    }
    initial_state["data"][0] = 1
    times = FundamentalTimeBasis(20000, 8e-10)
    states = solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )
    probabilities = from_state_vector_list(states)

    isf = calculate_isf_approximate_locations(
        from_state_vector(initial_state),
        average_probabilities(probabilities, (0,)),
        get_jianding_isf_112bar(),
    )
    fig, _, _ = plot_isf_against_time(isf)
    fig.show()
    input()


def get_repeat_average_isf(
    isf: SingleBasisDiagonalOperator[StackedBasisLike[FundamentalBasis[Any], _B0]],
) -> StatisticalDiagonalOperator[_B0, _B0]:
    average = average_eigenvalues(isf, (0,))
    standard_deviation = np.std(isf["data"].reshape(isf["basis"][0].shape), axis=0)
    standard_deviation[0] = standard_deviation[1]
    standard_deviation /= np.sqrt(isf["basis"][0].shape[0])
    return {
        "basis": StackedBasis(average["basis"][0][0], average["basis"][1][0]),
        "data": average["data"],
        "standard_deviation": standard_deviation,
    }


def get_lambda_values(
    temperature: float, n: int = 2
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    energies = np.array([get_hydrogen_energy_difference(0, i) for i in range(n)])
    return np.exp(-energies / (Boltzmann * temperature))


def _get_average_repeat_average_isf(
    temperature: float, band: int, times: _B0, repeats: int
) -> StatisticalDiagonalOperator[_B0, _B0]:
    isf = list[StatisticalDiagonalOperator[_B0, _B0]]()
    for repeat in range(repeats):
        probabilities_i = from_state_vector_list(
            get_simulation_at_temperature(temperature, band, times, _i=repeat)
        )
        isf_i = get_repeat_average_isf(
            calculate_isf_approximate_locations(
                get_probability(probabilities_i, 0),
                probabilities_i,
                get_jianding_isf_112bar(),
            )
        )
        assert np.count_nonzero(np.logical_not(np.isfinite(isf_i["data"]))) == 0
        isf.append(isf_i)

    return {
        "basis": isf[0]["basis"],
        "data": np.average([i["data"] for i in isf], axis=0),
        "standard_deviation": np.average([i["standard_deviation"] for i in isf], axis=0)
        / np.sqrt(repeats),
    }


def get_average_simulation_isf(
    temperature: float, times: _B0
) -> StatisticalDiagonalOperator[_B0, _B0]:
    isf_s = [
        _get_average_repeat_average_isf(temperature, 0, times, repeats=6),
        _get_average_repeat_average_isf(temperature, 1, times, repeats=6),
        _get_average_repeat_average_isf(temperature, 2, times, repeats=2),
        _get_average_repeat_average_isf(temperature, 3, times, repeats=2),
        _get_average_repeat_average_isf(temperature, 4, times, repeats=2),
        _get_average_repeat_average_isf(temperature, 5, times, repeats=2),
        _get_average_repeat_average_isf(temperature, 6, times, repeats=2),
        _get_average_repeat_average_isf(temperature, 7, times, repeats=2),
    ]

    values = get_lambda_values(temperature, 8)
    probabilites = values / np.sum(values)
    return {
        "basis": isf_s[0]["basis"],
        "data": np.average(
            [isf["data"] for isf in isf_s],
            axis=0,
            weights=probabilites,
        ),
        "standard_deviation": np.average(
            [isf["standard_deviation"] for isf in isf_s],
            axis=0,
            weights=probabilites,
        ),
    }


def plot_average_isf_all_temperatures() -> None:
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    times = [
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 22e-10),
        # Fine ..
        EvenlySpacedTimeBasis(2000, 20, 0, 12e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 6e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 6e-10),
    ]
    valid_times = [8.0e-10, 3.8e-10, 3.0e-10, 1.1e-10, 1.1e-10, 6e-11, 3.5e-11]
    for temperature, time, start in reversed(
        list(zip(temperatures, times, valid_times, strict=True))
    ):
        isf = get_average_simulation_isf(temperature, time)
        assert np.all(np.isfinite(isf["data"]))
        assert np.all(np.isfinite(isf["standard_deviation"]))
        fig, ax, _ = plot_isf_against_time(isf)

        fit = fit_isf_to_fey_4_variable_model_110(
            isf, get_lambda_values(temperature)[1], start_t=start
        )
        plot_isf_fey_4_variable_fit_against_time(fit, isf["basis"][0].times, ax=ax)
        probabilities_i = from_state_vector_list(
            get_repeated_simulation_at_temperature(temperature, 0, time, repeats=6)
        )
        fig, ax2, lines = plot_average_probability_per_band(
            probabilities_i, ax=ax.twinx()
        )
        plot_expected_occupation_per_band_hydrogen(temperature, ax=ax2)
        for line in lines:
            line.set_linestyle("--")
        fig.show()
        input()
    input()


def plot_rate_against_temperature() -> None:
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    times = [
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 22e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 12e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 6e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 6e-10),
    ]
    start_times = [8.0e-10, 3.8e-10, 3.0e-10, 1.1e-10, 1.1e-10, 6e-11, 3.5e-11]
    fits = list[ISFFey4VariableFit]()
    for temperature, time, start_t in list(
        zip(temperatures, times, start_times, strict=True)
    ):
        isf = get_average_simulation_isf(temperature, time)
        fit = fit_isf_to_fey_4_variable_model_110(
            isf, get_lambda_values(temperature)[1], start_t=start_t
        )
        fits.append(fit)

    fig, ax = plt.subplots()

    ax.plot(temperatures, [fit.fast_rate for fit in fits])
    ax.plot(temperatures, [fit.slow_rate for fit in fits])
    data = get_experiment_data()
    ax.errorbar(
        data["temperature"],
        data["rate"],
        yerr=[data["rate"] - data["lower_error"], data["upper_error"] - data["rate"]],
    )

    scale = FuncScale(ax, [lambda x: 1 / x, lambda x: 1 / x])  # type: ignore unknown
    ax.set_xscale(scale)
    ax.set_yscale("log")
    fig.show()
    input()


def get_average_initial_site_probability(
    temperature: float, band: int, times: _B0, repeats: int
) -> ProbabilityVectorList[_B0, FundamentalBasis[Literal[2]]]:
    probabilities = np.zeros((0, times.times.size, 800), dtype=np.float64)
    for repeat in range(repeats):
        probabilities_i = from_state_vector_list(
            get_simulation_at_temperature(temperature, band, times, _i=repeat)
        )
        probabilities = np.append(
            probabilities,
            probabilities_i["data"].reshape(*probabilities_i["basis"][0].shape, -1),
            axis=0,
        )

    return {"basis": StackedBasis(times, FundamentalBasis(2)), "data": data}


def plot_initial_site_occupation() -> None:
    temperatures = np.array([100, 125, 150, 175, 200, 225, 250])
    times = [
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 40, 0, 44e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 22e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 12e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 6e-10),
        EvenlySpacedTimeBasis(2000, 20, 0, 6e-10),
    ]
    for temperature, time in list(zip(temperatures, times, strict=True)):
        probability = get_average_initial_site_probability(temperature, 0, time, 3)

        fig, _, _ = plot_probability_against_time(probability)
        fig.show()


def plot_ballistic_isf_against_time() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((10, 10), 8, 150)
    hamiltonian = get_coherent_hamiltonian(a_matrix["basis"][0])

    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"][0],
        "data": np.zeros(a_matrix["basis"].shape[0], dtype=np.complex128),
    }
    initial_state["data"][0] = 1
    times = FundamentalTimeBasis(20000, 1e-15)

    states = solve_schrodinger_equation(initial_state, times, hamiltonian)

    probabilities = from_state_vector_list(states)
    isf = calculate_isf_approximate_locations(
        from_state_vector(initial_state),
        probabilities,
        get_jianding_isf_112bar(),
    )
    fig, _, _ = plot_isf_against_time(isf)
    fig.show()
    input()
