from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from surface_potential_analysis.dynamics.incoherent_propagation.eigenstates import (
    calculate_equilibrium_state,
    calculate_tunnelling_simulation_state,
)
from surface_potential_analysis.dynamics.incoherent_propagation.plot import (
    plot_occupation_per_band,
    plot_occupation_per_site,
)
from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
    TunnellingAMatrix,
    get_tunnelling_a_matrix_from_function,
    get_tunnelling_m_matrix,
)
from surface_potential_analysis.util.decorators import npy_cached

from hydrogen_nickel_111.s6_a_calculation import (
    a_function_deuterium,
    a_function_hydrogen,
)
from hydrogen_nickel_111.surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import FundamentalAxis
    from surface_potential_analysis.operator.operator import DiagonalOperator

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)


def _get_get_tunnelling_a_matrix_hydrogen_cache(
    shape: tuple[_L0Inv, _L1Inv], n_bands: _L2Inv, temperature: float
) -> Path:
    return get_data_path(
        f"dynamics/a_matrix_hydrogen_{shape[0]}_{shape[1]}_{n_bands}_{temperature}k.npy"
    )


@npy_cached(_get_get_tunnelling_a_matrix_hydrogen_cache, load_pickle=True)  # type: ignore[misc]
def get_tunnelling_a_matrix_hydrogen(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    temperature: float,
) -> TunnellingAMatrix[
    tuple[
        FundamentalAxis[_L0Inv],
        FundamentalAxis[_L1Inv],
        FundamentalAxis[_L2Inv],
    ]
]:
    def a_function(
        i: int, j: int, offset_i: tuple[int, int], offset_j: tuple[int, int]
    ) -> float:
        return a_function_hydrogen(i, j, offset_i, offset_j, temperature)

    return get_tunnelling_a_matrix_from_function(shape, n_bands, a_function)


def _get_get_tunnelling_a_matrix_deuterium_cache(
    shape: tuple[_L0Inv, _L1Inv], n_bands: _L2Inv, temperature: float
) -> Path:
    return get_data_path(
        f"dynamics/a_matrix_deuterium_{shape[0]}_{shape[1]}_{n_bands}_{temperature}k.npy"
    )


@npy_cached(_get_get_tunnelling_a_matrix_deuterium_cache, load_pickle=True)  # type: ignore[misc]
def get_tunnelling_a_matrix_deuterium(
    shape: tuple[_L0Inv, _L1Inv],
    n_bands: _L2Inv,
    temperature: float,
) -> TunnellingAMatrix[
    tuple[
        FundamentalAxis[_L0Inv],
        FundamentalAxis[_L1Inv],
        FundamentalAxis[_L2Inv],
    ]
]:
    def a_function(
        i: int, j: int, offset_i: tuple[int, int], offset_j: tuple[int, int]
    ) -> float:
        return a_function_deuterium(i, j, offset_i, offset_j, temperature)

    return get_tunnelling_a_matrix_from_function(shape, n_bands, a_function)


def test_normalization_of_m_matrix() -> None:
    rng = np.random.default_rng()
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)

    initial = rng.random(m_matrix["array"].shape[0])
    initial /= np.linalg.norm(initial)
    actual = np.sum(np.tensordot(m_matrix["array"], initial, (1, 0)))  # type: ignore[var-annotated]
    np.testing.assert_array_almost_equal(0, actual)

    np.testing.assert_array_almost_equal(0, np.sum(m_matrix["array"], axis=0))

    np.testing.assert_array_equal(1, np.diag(m_matrix["array"]) <= 0)
    np.testing.assert_array_equal(1, a_matrix["array"] >= 0)


def get_equilibrium_state_on_surface() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    state = calculate_equilibrium_state(m_matrix)
    print(state["vector"])  # noqa: T201
    print(np.sum(state["vector"]))  # noqa: T201


def plot_occupation_on_surface() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state: DiagonalOperator[Any, Any] = {
        "basis": m_matrix["basis"],
        "dual_basis": m_matrix["basis"],
        "vector": np.zeros(m_matrix["array"].shape[0]),
    }
    initial_state["vector"][0] = 1
    times = np.linspace(0, 9e-10, 1000)
    state = calculate_tunnelling_simulation_state(m_matrix, initial_state, times)

    fig, ax = plot_occupation_per_band(state, times)
    fig.show()

    fig, ax = plot_occupation_per_site(state, times)
    fig.show()
    input()


def get_simulated_state_on_surface() -> None:
    a_matrix = get_tunnelling_a_matrix_hydrogen((5, 5), 2, 150)
    m_matrix = get_tunnelling_m_matrix(a_matrix)
    initial_state: DiagonalOperator[Any, Any] = {
        "basis": m_matrix["basis"],
        "dual_basis": m_matrix["basis"],
        "vector": np.zeros(m_matrix["array"].shape[0]),
    }
    initial_state["vector"][0] = 1
    state = calculate_tunnelling_simulation_state(
        m_matrix, initial_state, np.array([99999])
    )
    print(state["vectors"])  # noqa: T201
    print(np.sum(state["vectors"]))  # noqa: T201
