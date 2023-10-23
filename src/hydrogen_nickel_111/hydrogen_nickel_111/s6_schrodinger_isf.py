from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import FuncScale
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.stacked_basis import StackedBasis
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
from surface_potential_analysis.dynamics.stochastic_schrodinger.solve import (
    get_simplified_collapse_operators_from_a_matrix,
    solve_stochastic_schrodinger_equation,
)
from surface_potential_analysis.operator.operator import (
    StatisticalDiagonalOperator,
    average_eigenvalues,
)
from surface_potential_analysis.probability_vector.probability_vector import (
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
from hydrogen_nickel_111.s6_schrodinger_dynamics import get_coherent_hamiltonian

from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.basis.basis import FundamentalBasis
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

_B0 = TypeVar("_B0", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def _sse_sim_cache(
    temperature: float, idx: int, times: Any, _i: int = 0  # noqa: ARG001,ANN401
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
        "data": np.zeros(a_matrix["basis"][0].n, dtype=np.complex_),
    }
    initial_state["data"][idx] = 1
    return solve_stochastic_schrodinger_equation(
        initial_state, times, hamiltonian, collapse_operators, n_trajectories=20
    )


def plot_average_isf_against_time() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((6, 6), 6, 150)
    collapse_operators = get_simplified_collapse_operators_from_a_matrix(a_matrix)

    hamiltonian = get_coherent_hamiltonian(a_matrix["basis"][0])
    hamiltonian["data"] = np.zeros_like(hamiltonian["data"])

    initial_state: StateVector[Any] = {
        "basis": a_matrix["basis"][0],
        "data": np.zeros(a_matrix["basis"].shape[0], dtype=np.complex_),
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
    isf: SingleBasisDiagonalOperator[StackedBasisLike[FundamentalBasis[Any], _B0]]
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


def get_lambda(temperature: float) -> np.float_:
    return np.exp(-get_hydrogen_energy_difference(0, 1) / (Boltzmann * temperature))


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
    isf_0 = _get_average_repeat_average_isf(temperature, 0, times, 2)
    isf_1 = _get_average_repeat_average_isf(temperature, 0, times, 2)
    p0 = 1 / (1 + get_lambda(temperature))
    return {
        "basis": isf_0["basis"],
        "data": p0 * isf_0["data"] + (1 - p0) * isf_1["data"],
        "standard_deviation": p0 * isf_0["standard_deviation"]
        + (1 - p0) * isf_1["standard_deviation"],
    }


def plot_average_isf_all_temperatures() -> None:
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
        isf = get_average_simulation_isf(temperature, time)
        fig, ax, _ = plot_isf_against_time(isf)

        fit = fit_isf_to_fey_4_variable_model_110(isf, get_lambda(temperature))
        plot_isf_fey_4_variable_fit_against_time(fit, isf["basis"][0].times, ax=ax)
        fig.show()
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
    fits = list[ISFFey4VariableFit]()
    for temperature, time in list(zip(temperatures, times, strict=True)):
        isf = get_average_simulation_isf(temperature, time)
        fit = fit_isf_to_fey_4_variable_model_110(isf, get_lambda(temperature))
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
