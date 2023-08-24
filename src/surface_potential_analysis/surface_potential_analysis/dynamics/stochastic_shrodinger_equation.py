from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import qutip
import scipy.sparse

from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_basis import (
    get_basis_from_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis.axis.axis import FundamentalAxis
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_basis import (
        TunnellingSimulationBandsAxis,
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_matrix import (
        TunnellingAMatrix,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector import (
        StateVector,
    )
    from surface_potential_analysis.wavepacket.wavepacket import (
        WavepacketWithEigenvalues,
    )

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])
    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)


def generate_hamiltonian_from_wavepackets(
    _wavepacket: WavepacketWithEigenvalues[Any, Any],
) -> SingleBasisOperator[TunnellingSimulationBasis[Any, Any, Any]]:
    """
    Given a set of wavepackets, calculate the relevant hamiltonian.

    Parameters
    ----------
    _wavepacket : Wavepacket[Any, Any]

    Returns
    -------
    SingleBasisOperator[TunnellingSimulationBasis[Any, Any, Any]]

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError


def get_collapse_operators_from_a_matrix(
    matrix: TunnellingAMatrix[_B0Inv],
) -> list[SingleBasisOperator[_B0Inv]]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_axis : TunnellingSimulationBandsAxis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalAxis[_L0Inv], FundamentalAxis[_L1Inv], TunnellingSimulationBandsAxis[_L2Inv]]]]
    """
    print(matrix["array"].shape)
    np.fill_diagonal(matrix["array"], 0)
    return [
        {
            "basis": matrix["basis"],
            "dual_basis": matrix["basis"],
            "array": scipy.sparse.coo_array(
                ([matrix["array"][idx]], ([np.int32(idx[0])], [np.int32(idx[1])])),
                shape=matrix["array"].shape,
            ),
        }
        for idx in zip(*np.nonzero(matrix["array"]), strict=True)
    ]


def get_collapse_operators_from_function(
    shape: tuple[_L0Inv, _L1Inv],
    bands_axis: TunnellingSimulationBandsAxis[_L2Inv],
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
        tuple[
            FundamentalAxis[_L0Inv],
            FundamentalAxis[_L1Inv],
            TunnellingSimulationBandsAxis[_L2Inv],
        ]
    ]
]:
    """
    Given a function which produces the collapse operators S_{i,j} calculate the relevant collapse operators.

    Parameters
    ----------
    shape : tuple[_L0Inv, _L1Inv]
    bands_axis : TunnellingSimulationBandsAxis[_L2Inv]
    a_function : Callable[ [ int, int, tuple[int, int], tuple[int, int], ], float, ]

    Returns
    -------
    list[SingleBasisOperator[ tuple[ FundamentalAxis[_L0Inv], FundamentalAxis[_L1Inv], TunnellingSimulationBandsAxis[_L2Inv]]]]
    """
    operators: list[
        SingleBasisOperator[
            tuple[
                FundamentalAxis[_L0Inv],
                FundamentalAxis[_L1Inv],
                TunnellingSimulationBandsAxis[_L2Inv],
            ]
        ]
    ] = []
    basis = get_basis_from_shape(shape, bands_axis)

    n_sites = np.prod(shape)
    n_bands = bands_axis.fundamental_n
    array = np.zeros((n_sites * n_bands, n_sites * n_bands))
    for i in range(array.shape[0]):
        for n1 in range(n_bands):
            for d1 in range(9):
                (i0, j0, n0) = np.unravel_index(i, (*shape, n_bands))
                d1_stacked = np.unravel_index(d1, (3, 3)) - np.array([1, 1])
                (i1, j1) = (i0 + d1_stacked[0], j0 + d1_stacked[1])
                j = np.ravel_multi_index((i1, j1, n1), (*shape, n_bands), mode="wrap")

                data = a_function(int(n0), n1, (0, 0), (d1_stacked[0], d1_stacked[1]))
                operators.append(
                    {
                        "basis": basis,
                        "dual_basis": basis,
                        "array": scipy.sparse.coo_array((data, (i, j)), shape=shape),
                    }
                )
    return operators


def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B0Inv],
    hamiltonian: SingleBasisOperator[_B0Inv],
    collapse_operators: list[SingleBasisOperator[_B0Inv]],
) -> None:
    hamiltonian_qobj = qutip.Qobj(hamiltonian["array"])
    initial_state_qobj = qutip.Qobj(initial_state["vector"])
    sc_ops = [qutip.Qobj(op["array"]) for op in collapse_operators[:2]]
    result = qutip.ssesolve(
        hamiltonian_qobj,
        initial_state_qobj,
        np.linspace(0, 1e-10, 1000),
        sc_ops=sc_ops,
        e_ops=[],
    )
    print(result.states)
    print(result.ntraj)
