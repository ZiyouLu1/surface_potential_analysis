from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.hamiltonian.hamiltonian import stack_hamiltonian
from surface_potential_analysis.interpolation import pad_ft_points

from .s2_hamiltonian import generate_hamiltonian_sho, generate_sho_basis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import ExplicitBasis, PositionBasis


def _normalize_sho_basis(
    basis: ExplicitBasis[int, PositionBasis[int]]
) -> ExplicitBasis[int, PositionBasis[int]]:
    turning_point = basis["vectors"][
        np.arange(basis["vectors"].shape[0]),
        np.argmax(
            np.abs(basis["vectors"][:, : basis["vectors"].shape[1] // 2]), axis=1
        ),
    ]

    normalized = np.exp(-1j * np.angle(turning_point))[:, np.newaxis] * basis["vectors"]
    return {"_type": "explicit", "parent": basis["parent"], "vectors": normalized}


def plot_sho_basis() -> None:
    """Plot the sho basis used for the nickel surface."""
    (infinate, finite) = generate_sho_basis(16)

    fig, ax = plt.subplots()

    normalized = _normalize_sho_basis(infinate)
    _, _, lines = plot_explicit_basis_states_x(normalized, ax=ax, measure="real")
    for line in lines:
        line.set_color("tab:blue")

    normalized = _normalize_sho_basis(finite)
    _, _, lines = plot_explicit_basis_states_x(normalized, ax=ax, measure="real")
    for line in lines:
        line.set_color("tab:orange")

    fig.show()
    input()


def test_hamiltonian_large_resolution() -> None:
    """
    Test the generated hamiltonian when using a larger basis.

    We expect the result should be equal if we throw away the additional states.
    """
    resolution0 = (23, 23, 12)
    h0 = generate_hamiltonian_sho(
        shape=(2 * resolution0[0], 2 * resolution0[1], 500),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution0,
    )
    h0_stacked = stack_hamiltonian(h0)

    h0_large = generate_hamiltonian_sho(
        shape=(10 * resolution0[0], 10 * resolution0[1], 500),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution0,
    )
    np.testing.assert_array_almost_equal(h0_large["array"], h0["array"])
    # Max absolute difference: 6.531858e-24
    # Max relative difference: 119.78769123
    # Max value 3.04e-18
    np.testing.assert_array_equal(h0_large["array"], h0["array"])
    resolution1 = (25, 25, 16)
    h1 = generate_hamiltonian_sho(
        shape=(2 * resolution1[0], 2 * resolution1[1], 500),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution1,
    )
    h1_stacked = stack_hamiltonian(h1)
    expected = pad_ft_points(
        h1_stacked["array"],  # type:ignore[arg-type]
        (resolution0[0], resolution0[0], resolution0[0], resolution0[0]),
        axes=(0, 1, 3, 4),
    )[:, :, : resolution0[2], :, :, : resolution0[2]]
    np.testing.assert_array_almost_equal(expected, h0_stacked["array"])
    # Max absolute difference: 8.18056993e-25
    # Max relative difference: 53.43269578
    np.testing.assert_array_equal(expected, h0_stacked["array"])


def test_hamiltonian_very_large_resolution() -> None:
    """
    Test the generated hamiltonian when using a larger basis.

    We expect the result should be equal if we throw away the additional states.
    """
    resolution = (23, 23, 12)
    h0 = generate_hamiltonian_sho(
        shape=(2 * resolution[0], 2 * resolution[1], 250),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution,
    )

    h1 = generate_hamiltonian_sho(
        shape=(200, 200, 250),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution,
    )
    np.testing.assert_array_almost_equal(h1["array"], h0["array"])
    # Max absolute difference: 6.531858e-24
    # Max relative difference: 119.78769123
    # Max value 3.04e-18

    h2 = generate_hamiltonian_sho(
        shape=(100, 100, 250),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution,
    )
    np.testing.assert_array_almost_equal(h1["array"], h2["array"])
    # Max absolute difference: 1.83756526e-24
    # Max relative difference: 13.57765139

    h3 = generate_hamiltonian_sho(
        shape=(250, 250, 250),
        bloch_phase=np.array([0, 0, 0]),
        resolution=resolution,
    )
    np.testing.assert_array_almost_equal(h1["array"], h3["array"])
    np.testing.assert_array_equal(h1["array"], h3["array"])
    # Max absolute difference: 3.62036857e-25
    # Max relative difference: 3.32098834
    # This is comparable to the size of the bandwidth (~1E-25)
    # And we would hope the errors cancel/ are not around the
    # minimum of energy in some way
