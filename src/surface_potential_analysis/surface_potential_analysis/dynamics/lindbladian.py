from __future__ import annotations

from typing import Generic, Literal, TypedDict, TypeVar

import numpy as np
import qutip
import scipy.linalg


class NonHermitianGamma(TypedDict):
    """
    Stores the transition coefficients from state l_1 to state l_0.

    Indexed such that for a density matrix with n states i_1' i_1
    array.reshape(n, n, n, n)[i_1' i_1, i_0' i_0].
    """

    array: np.ndarray[tuple[int, int], np.dtype[np.complex_]]


class JumpOperatorCollection(TypedDict):
    """
    Stores the transition coefficients from state l_1 to state l_0.

    Indexed such that for a density matrix with n states i_1' i_1
    array.reshape(n, n, n, n)[i_1' i_1, i_0' i_0].
    """

    arrays: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]


_L0Inv = TypeVar("_L0Inv", bound=int)


class NonHermitianGammaCoefficientMatrix(TypedDict, Generic[_L0Inv]):
    """
    Represents the matrix of gamma coefficients on a surfaces.

    The coefficients np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    represent the total coefficient G[i,j,dx] from state of type i to j in a unit cell
    with an offset of dx at the location i.

    dx is indexed such that np.ravel_multi_index(offset, (3,3), mode="wrap") gives the flat index
    for offsets (-1/+0/+1, -1/+0/+1)
    """

    array: np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]


def calculate_gamma_two_state(
    shape: tuple[int, int],
    coefficient_matrix: NonHermitianGammaCoefficientMatrix[Literal[2]],
) -> NonHermitianGamma:
    """Given a coefficient matrix, calculate the non-hermitian gamma matrix."""
    n_sites = 2 * np.product(shape)

    gamma_stacked = np.zeros((n_sites, n_sites))

    for from_idx in range(n_sites):
        for offset_0 in [-1, 0, 1]:
            for offset_1 in [-1, 0, 1]:
                for to_ty in range(2):
                    from_ty, i, j = np.unravel_index(from_idx, (2, *shape))
                    i1 = (i + offset_0) % shape[0]
                    j1 = (j + offset_1) % shape[1]
                    to_idx = np.ravel_multi_index(
                        [to_ty, i1, j1], (2, *shape), mode="wrap"
                    )
                    jump_idx = np.ravel_multi_index(
                        (offset_0, offset_1), (3, 3), mode="wrap"
                    )
                    gamma_stacked[from_idx, to_idx] = coefficient_matrix["array"][
                        from_ty, to_ty, jump_idx
                    ]

    return {"array": gamma_stacked.reshape(2 * n_sites)}


def calculate_jump_operators(gamma: NonHermitianGamma) -> JumpOperatorCollection:
    """
    Given gamma, calculate the diagonal jump operators.

    Parameters
    ----------
    gamma : NonHermitianGamma

    Returns
    -------
    JumpOperatorCollection
    """
    energies, vectors = scipy.linalg.eig(gamma["array"])

    arrays = np.array(
        [
            np.sqrt(e) * v[:, np.newaxis] * np.conj(v[np.newaxis, :])
            for (e, v) in zip(energies, vectors.T, strict=True)
        ]
    )

    return {"arrays": arrays}


def solve_master_equation(jump_operators: JumpOperatorCollection) -> qutip.Result:
    """
    Solve the quantum master equation for a given set of jump operators.

    NOTE: this works in the interaction picture, ie H = 0

    Parameters
    ----------
    jump_operators : JumpOperatorCollection
        _description_

    Returns
    -------
    qutip.Result
    """
    n_states = jump_operators["arrays"].shape[1]
    rho_0_matrix = np.zeros(shape=(n_states, n_states))
    rho_0_matrix[0, 0] = 1
    rho_0 = qutip.Qobj(rho_0_matrix, type="operator")
    c_ops = [
        qutip.Qobj(c_op_matrix, type="operator")
        for c_op_matrix in jump_operators["arrays"]
    ]
    hamiltonian_matrix = np.zeros(shape=(n_states, n_states))
    hamiltonian = qutip.Qobj(
        hamiltonian_matrix, hamiltonian_matrix.shape, type="operator"
    )
    return qutip.mesolve(hamiltonian, rho0=rho_0, c_ops=c_ops)
