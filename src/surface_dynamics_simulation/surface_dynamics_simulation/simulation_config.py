from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar

import numpy as np
import scipy
import scipy.linalg
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from collections.abc import Callable

_LCov = TypeVar("_LCov", bound=int)

_NYInv = TypeVar("_NYInv", bound=int)
_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


class SimulationConfig(TypedDict, Generic[_LCov]):
    """Dict to store coefficients required for simulation."""

    shape: tuple[int, int]
    """
    The number of repeats of the unit cell.
    """
    # TODO: complex??
    coherent_coefficients: np.ndarray[
        tuple[_LCov, Literal[3], Literal[3]], np.dtype[np.float_]
    ]
    incoherent_coefficients: np.ndarray[tuple[_LCov, _LCov], np.dtype[np.complex_]]


def _get_solver_function(
    config: SimulationConfig[_L0Inv],
) -> Callable[
    [np.float_, np.ndarray[tuple[_NYInv], np.dtype[np.complex_]]],
    np.ndarray[tuple[_NYInv], np.dtype[np.complex_]],
]:
    def fun(
        _t: np.float_, y: np.ndarray[tuple[_NYInv], np.dtype[np.complex_]]
    ) -> np.ndarray[tuple[_NYInv], np.dtype[np.complex_]]:
        """Return dy/dt for the solver given y."""
        stacked = y.reshape(*config["shape"], config["coherent_coefficients"].shape[0])
        stacked += (
            1j
            * config["coherent_coefficients"][np.newaxis, np.newaxis, :, 0, 0]
            * stacked
        )
        return stacked.reshape(-1)

    return fun


def build_tunnelling_matrix(
    coefficients: np.ndarray[tuple[_L0Inv, _L0Inv, Literal[4]], np.dtype[np.float_]],
    shape: tuple[int, int],
) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
    """
    Given a list of hopping coefficients build the tunnelling matrix for a grid of the given shape.

    Parameters
    ----------
    coefficients : np.ndarray[tuple[_L0Inv, _L0Inv, Literal[8]], np.dtype[np.float_]]
    shape : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float_]]
    """
    n_states = coefficients.shape[0]
    out = np.zeros((np.prod(shape), n_states, np.prod(shape), n_states))

    for ix in range(out.shape[0]):
        for dix0 in [0, 1]:
            for dix1 in [0, 1]:
                ix0, ix1 = np.unravel_index(ix, shape)
                jx = np.ravel_multi_index((ix0 + dix0, ix1 + dix1), shape, mode="wrap")
                hop_idx = np.ravel_multi_index((dix0, dix1), (2, 2), mode="wrap")
                print(ix, jx, coefficients[:, :, hop_idx])  # noqa: T201
                # TODO: not += ??
                out[ix, :, jx, :] += coefficients[:, :, hop_idx]

    return out.reshape(np.prod(shape) * n_states, np.prod(shape) * n_states)  # type: ignore[no-any-return]


def simulate_tunnelling(  # noqa: D103
    matrix: np.ndarray[tuple[_L0Inv, _L0Inv], np.dtype[np.float_]],
    initial_state: np.ndarray[tuple[_L0Inv], np.dtype[np.float_]],
    times: np.ndarray[tuple[_L1Inv], np.dtype[np.float_]],
) -> np.ndarray[tuple[_L0Inv, _L1Inv], np.dtype[np.float_]]:
    print("is hermitian", scipy.linalg.ishermitian(matrix))  # noqa: T201
    print("is symmetric", scipy.linalg.issymmetric(matrix))  # noqa: T201
    energies, vectors = scipy.linalg.eigh(matrix)
    print(energies.shape, vectors.shape)  # noqa: T201

    solved = scipy.linalg.solve(vectors, initial_state)
    print(solved.shape)  # noqa: T201
    for i, e in enumerate(energies):
        print(i, e, vectors[:, i])  # noqa: T201

    constants = solved[:, np.newaxis] * np.exp(
        energies[:, np.newaxis] * times[np.newaxis, :]
    )
    print(constants)  # noqa: T201

    return np.sum(vectors[:, :, np.newaxis] * constants[np.newaxis], axis=1)  # type: ignore[no-any-return]


if __name__ == "__main__":
    config: SimulationConfig[Literal[1]] = {
        "coherent_coefficients": np.array([[[1e-8, 0, 0], [0, 0, 0], [0, 0, 0]]]),
        "incoherent_coefficients": np.array([[0]]),
        "shape": (1, 1),
    }
    y0 = np.zeros(
        (*config["shape"], config["coherent_coefficients"].shape[0]), dtype=complex
    )
    y0[0, 0, 0] = 1

    out = solve_ivp(
        _get_solver_function(config),
        t_span=(0, 1),
        method="RK45",
        y0=y0.flatten(),
    )

    out = solve_ivp(
        lambda _t, _y: 0.000001j,
        t_span=(0, 1),
        method="RK45",
        y0=[0 + 1j],
    )

    fig, ax = plt.subplots()
    (line,) = ax.plot(np.real(out.y[0, :]), np.imag(out.y[0, :]))
    (line,) = ax.twinx().twiny().plot(out.t, np.abs(out.y[0, :]))
    line.set_marker("x")
    fig.show()
    input()
