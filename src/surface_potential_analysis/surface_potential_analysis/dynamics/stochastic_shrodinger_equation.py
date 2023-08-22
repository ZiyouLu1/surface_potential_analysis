from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import qutip

if TYPE_CHECKING:
    from surface_potential_analysis.dynamics.incoherent_propagation.tunnelling_basis import (
        TunnellingSimulationBasis,
    )
    from surface_potential_analysis.operator.operator import SingleBasisOperator
    from surface_potential_analysis.state_vector import (
        StateVector,
    )

    _B0Inv = TypeVar("_B0Inv", bound=TunnellingSimulationBasis[Any, Any, Any])


def generate_hamiltonian_from_wavepackets() -> (
    SingleBasisOperator[TunnellingSimulationBasis[Any, Any, Any]]
):
    return {"basis": basis, "dual_basis": basis, "array": array}


def solve_stochastic_schrodinger_equation(
    initial_state: StateVector[_B0Inv],
    hamiltonian: SingleBasisOperator[_B0Inv],
    collapse_operators: list[SingleBasisOperator[_B0Inv]],
) -> None:
    hamiltonian_qobj = qutip.Qobj(hamiltonian["array"])
    initial_state_qobj = qutip.Qobj(initial_state["vector"])
    sc_ops = [qutip.Qobj(op["array"]) for op in collapse_operators]
    result = qutip.ssesolve(
        hamiltonian_qobj, initial_state_qobj, [0, 1e-10], sc_ops=sc_ops, e_ops=[]
    )
    print(result.states)
    print(result.ntraj)
