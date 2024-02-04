from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import qutip
import qutip.ui
import scipy.sparse

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.tunnelling_basis import (
    get_basis_from_shape,
)
from surface_potential_analysis.dynamics.util import build_hop_operator, get_hop_shift

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
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
                hop_val = jump_array[n_0, hop_shift[0], hop_shift[1], n_1]
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
    initial_state: StateVector[_B0Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B0Inv],
    collapse_operators: list[SingleBasisOperator[_B0Inv]],
    *,
    n_trajectories: _L1Inv,
    combine_collapse_operators: bool = False,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B0Inv]:
    ...


@overload
def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B0Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B0Inv],
    collapse_operators: list[SingleBasisOperator[_B0Inv]],
    *,
    n_trajectories: Literal[1] = 1,
    combine_collapse_operators: bool = False,
) -> StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B0Inv]:
    ...


def solve_stochastic_schrodinger_equation(  # type: ignore bad overload
    initial_state: StateVector[_B0Inv],
    times: _AX0Inv,
    hamiltonian: SingleBasisOperator[_B0Inv],
    collapse_operators: list[SingleBasisOperator[_B0Inv]],
    *,
    n_trajectories: _L1Inv | Literal[1] = 1,
) -> (
    StateVectorList[StackedBasisLike[FundamentalBasis[Literal[1]], _AX0Inv], _B0Inv]
    | StateVectorList[StackedBasisLike[FundamentalBasis[_L1Inv], _AX0Inv], _B0Inv]
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
    hamiltonian_qobj = qutip.Qobj(
        hamiltonian["data"].reshape(hamiltonian["basis"].shape)
    ).to("CSR")
    initial_state_qobj = qutip.Qobj(initial_state["data"])

    sc_ops = [
        qutip.Qobj(op["data"].reshape(op["basis"].shape)).to("CSR")
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
            "method": "euler",
            "progress_bar": "enhanced",
            "store_states": True,
            "keep_runs_results": True,
            "map": "parallel",
            "dt": times.delta_t / times.fundamental_n,
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
        ).reshape(n_trajectories * 5, -1),
    }
