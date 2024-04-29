from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import qutip
import qutip.ui
import scipy.sparse
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.tunnelling_basis import (
    get_basis_from_shape,
)
from surface_potential_analysis.dynamics.util import build_hop_operator, get_hop_shift
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products,
    get_state_along_axis,
    get_state_vector,
    get_weighted_state_vector,
)

with contextlib.suppress(ImportError):
    from sse_solver_py import solve_sse_euler

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
        TunnellingAMatrix,
    )
    from surface_potential_analysis.dynamics.tunnelling_basis import (
        TunnellingSimulationBandsBasis,
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])
    _B1Inv = TypeVar("_B1Inv", bound=BasisLike[Any, Any])
    _B2Inv = TypeVar("_B2Inv", bound=BasisLike[Any, Any])
    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def get_collapse_operators_from_a_matrix(
    matrix: TunnellingAMatrix[_B0Inv],
) -> list[SingleBasisOperator[_B0Inv]]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]]]
    """
    data = matrix["data"].reshape(matrix["basis"].shape)
    np.fill_diagonal(data, 0)
    return [
        {
            "basis": matrix["basis"],
            "data": np.array(
                scipy.sparse.coo_array(
                    ([data[idx]], ([np.int32(idx[0])], [np.int32(idx[1])])),  # type: ignore always idx[1]
                    shape=data.shape,
                ).toarray(),
                dtype=np.float64,
            ),
        }
        for idx in zip(*np.nonzero(data), strict=True)
    ]


def get_simplified_collapse_operators_from_a_matrix(
    matrix: TunnellingAMatrix[_B0Inv], *, factor: float = 1
) -> list[SingleBasisOperator[_B0Inv]]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]]]
    """
    util = BasisUtil(matrix["basis"][0])
    (n_x1, n_x2, n_bands) = util.shape
    jump_array = matrix["data"].reshape(*util.shape, *util.shape)[0, 0]
    out: list[SingleBasisOperator[_B0Inv]] = []
    for n_0 in range(n_bands):
        for n_1 in range(n_bands):
            if n_0 == n_1:
                continue
            for hop in range(9):
                hop_shift = get_hop_shift(hop, 2)
                hop_val = factor * jump_array[n_0, hop_shift[0], hop_shift[1], n_1]
                if hop_val < 1:
                    continue

                operator = np.sqrt(hop_val) * build_hop_operator(hop, (n_x1, n_x2))
                array = np.zeros((*util.shape, *util.shape), dtype=np.complex128)
                array[:, :, n_1, :, :, n_0] = operator
                out.append({"basis": matrix["basis"], "data": array.reshape(-1)})

    return out


def get_collapse_operators_from_function(
    shape: tuple[_L0Inv, _L1Inv],
    bands_basis: TunnellingSimulationBandsBasis[_L2Inv],
    a_function: Callable[
        [
            int,
            int,
            tuple[int, int],
            tuple[int, int],
        ],
        float,
    ],
) -> list[
    SingleBasisOperator[
        StackedBasisLike[
            FundamentalBasis[_L0Inv],
            FundamentalBasis[_L1Inv],
            TunnellingSimulationBandsBasis[_L2Inv],
        ]
    ]
]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_basis : TunnellingSimulationBandsBasis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalBasis[_L0Inv], FundamentalBasis[_L1Inv], TunnellingSimulationBandsBasis[_L2Inv]]]]
    """
    operators: list[
        SingleBasisOperator[
            StackedBasisLike[
                FundamentalBasis[_L0Inv],
                FundamentalBasis[_L1Inv],
                TunnellingSimulationBandsBasis[_L2Inv],
            ]
        ]
    ] = []
    n_sites = np.prod(shape).item()
    n_bands = bands_basis.fundamental_n
    basis = get_basis_from_shape(shape, n_bands, bands_basis)

    array = np.zeros((n_sites * n_bands, n_sites * n_bands))
    for i in range(array.shape[0]):
        for n1 in range(n_bands):
            for d1 in range(9):
                (i0, j0, n0) = np.unravel_index(i, (*shape, n_bands))
                d1_stacked = np.unravel_index(d1, (3, 3)) - np.array([1, 1])
                (i1, j1) = (i0 + d1_stacked[0], j0 + d1_stacked[1])
                j = np.ravel_multi_index((i1, j1, n1), (*shape, n_bands), mode="wrap")

                data = np.zeros(shape, dtype=np.complex128)
                data[i, j] = a_function(
                    int(n0), n1, (0, 0), (d1_stacked[0], d1_stacked[1])
                )
                # TODO: use a single point basis, and a normal np.array...
                operators.append(
                    {"basis": StackedBasis(basis, basis), "data": data.reshape(-1)}
                )
    return operators


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]:
    ...


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]:
    ...


def solve_stochastic_schrodinger_equation(  # type: ignore bad overload
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
) -> (
    StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]
    | StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]
):
    """
    Given an initial state, use the stochastic schrodinger equation to solve the dynamics of the system.

    Parameters
    ----------
    initial_state : StateVector[_B0Inv]
    times : np.ndarray[tuple[int], np.dtype[np.float_]]
    hamiltonian : SingleBasisOperator[_B0Inv]
    collapse_operators : list[SingleBasisOperator[_B0Inv]]

    Returns
    -------
    StateVectorList[_B0Inv, _L0Inv]
    """
    if collapse_operators is None:
        collapse_operators = []
    hamiltonian_qobj = qutip.Qobj(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape) / hbar
    ).to("CSR")
    initial_state_qobj = qutip.Qobj(initial_state["data"])

    sc_ops = [
        qutip.Qobj(op["data"].reshape(op["basis"].shape) / hbar).to("CSR")
        for op in collapse_operators
    ]
    result = qutip.ssesolve(
        hamiltonian_qobj,
        initial_state_qobj,
        times.times,
        sc_ops=sc_ops,
        e_ops=[],
        options={
            # No other scheme scales well enough to such a large number of heatbath modes
            # "method": "euler",
            "progress_bar": "enhanced",
            "store_states": True,
            "keep_runs_results": True,
            "map": "parallel",
            "num_cpus": n_trajectories,
            "dt": times.fundamental_dt,
        },
        ntraj=n_trajectories,  # cspell:disable-line
    )
    return {
        "basis": StackedBasis(
            StackedBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": np.array(
            [
                np.asarray([state.full().reshape(-1) for state in trajectory])  # type: ignore unknown
                for trajectory in result.states  # type: ignore unknown
            ],
            dtype=np.complex128,
        ).reshape(-1),
    }


@overload
def solve_stochastic_schrodinger_equation_rust(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]:
    ...


@overload
def solve_stochastic_schrodinger_equation_rust(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]:
    ...


def solve_stochastic_schrodinger_equation_rust(  # type: ignore bad overload
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
) -> (
    StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]
    | StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]
):
    """
    Given an initial state, use the stochastic schrodinger equation to solve the dynamics of the system.

    Parameters
    ----------
    initial_state : StateVector[_B0Inv]
    times : np.ndarray[tuple[int], np.dtype[np.float_]]
    hamiltonian : SingleBasisOperator[_B0Inv]
    collapse_operators : list[SingleBasisOperator[_B0Inv]]

    Returns
    -------
    StateVectorList[_B0Inv, _L0Inv]
    """
    data = np.zeros(
        (n_trajectories, times.n, initial_state["data"].size), dtype=np.complex128
    )
    collapse_operators = [] if collapse_operators is None else collapse_operators

    for i in range(n_trajectories):
        out = solve_sse_euler(
            list(initial_state["data"]),
            [
                list(x / hbar)
                for x in hamiltonian["data"].reshape(hamiltonian["basis"].shape)
            ],
            [
                [list(x / np.sqrt(hbar)) for x in o["data"].reshape(o["basis"].shape)]
                for o in collapse_operators
            ],
            times.n,
            times.step,
            times.fundamental_dt,
        )
        data[i] = np.array(out).reshape(times.n, -1)

    return {
        "basis": StackedBasis(
            StackedBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": data.ravel(),
    }


rng = np.random.default_rng()


def _select_random_localized_state(
    states: StateVectorList[_B2Inv, _B1Inv],
) -> StateVector[_B1Inv]:
    """
    Select a random state built from states.

    This finds the states which diagonalize the overall density matrix,
    and selects states according to their overall occupation.

    Parameters
    ----------
    states : StateVectorList[_B2Inv, _B1Inv]

    Returns
    -------
    StateVector[_B1Inv]
    """
    states["data"].reshape(states["basis"].shape)

    op = calculate_inner_products(states, states)
    op["data"] /= np.linalg.norm(op["data"])
    # Vectors representing the combination of states that diagonalizes rho
    eigenstates = calculate_eigenvectors_hermitian(op)

    probabilities = eigenstates["eigenvalue"]
    probabilities /= np.sum(probabilities)

    idx = rng.choice(probabilities.size, p=probabilities)
    transformation = get_state_vector(eigenstates, idx)
    transformation["data"] /= np.sqrt(2)
    return get_weighted_state_vector(states, transformation)


@overload
def solve_stochastic_schrodinger_equation_localized(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: _L1Inv,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]:
    ...


@overload
def solve_stochastic_schrodinger_equation_localized(
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: Literal[1] = 1,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]:
    ...


def solve_stochastic_schrodinger_equation_localized(  # type: ignore bad overload
    initial_state: StateVector[_B1Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B1Inv],
    collapse_operators: list[SingleBasisOperator[_B1Inv]] | None = None,
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
    n_realizations: int = 2,
) -> (
    StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B1Inv]
    | StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B1Inv]
):
    """
    Find the quantum trajectores, using the localized stochastic schrodinger approach.

    Returns
    -------
    StateVectorList[StackedBasisLike[FundamentalBasis[int], _AX0Inv], _B1Inv]
    """
    data = np.zeros((n_trajectories, times.n, initial_state["basis"].n), dtype=complex)

    for trajectory in range(n_trajectories):
        state = initial_state
        for t in range(times.n):
            result = solve_stochastic_schrodinger_equation(
                state,
                EvenlySpacedTimeBasis(2, times.step, 0, times.dt),
                hamiltonian,
                collapse_operators,
                n_trajectories=n_realizations,
            )
            # Re-localize our state
            state = _select_random_localized_state(
                get_state_along_axis(result, axes=(1,), idx=(1,))
            )

            data[trajectory, t] = state["data"]

    return {
        "basis": StackedBasis(
            StackedBasis(FundamentalBasis(n_trajectories), times),
            hamiltonian["basis"][0],
        ),
        "data": data.reshape(-1),
    }
