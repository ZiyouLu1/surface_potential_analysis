from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.axis.axis import ExplicitAxis3d
from surface_potential_analysis.axis.plot import plot_explicit_basis_states_x
from surface_potential_analysis.basis.potential_basis import select_minimum_potential_3d
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
)
from surface_potential_analysis.util.interpolation import pad_ft_points

from hydrogen_nickel_111.s1_potential import get_interpolated_potential

from .s2_hamiltonian import (
    generate_sho_basis,
    get_hamiltonian_deuterium,
    get_hamiltonian_hydrogen_sho,
)


def _normalize_sho_basis(basis: ExplicitAxis3d[int, int]) -> ExplicitAxis3d[int, int]:
    turning_point = basis.vectors[
        np.arange(basis.vectors.shape[0]),
        np.argmax(np.abs(basis.vectors[:, : basis.vectors.shape[1] // 2]), axis=1),
    ]

    normalized = np.exp(-1j * np.angle(turning_point))[:, np.newaxis] * basis.vectors
    return ExplicitAxis3d(basis.delta_x, normalized)


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


def plot_deuterium_basis() -> None:
    shape = (50, 50, 100)
    hamiltonian = get_hamiltonian_deuterium(
        shape=shape,
        bloch_fraction=np.array([0, 0, 0]),
        resolution=(2, 2, 12),
    )
    fig, ax, _ = plot_explicit_basis_states_x(hamiltonian["basis"][2])

    potential = get_interpolated_potential(shape)
    minimum = select_minimum_potential_3d(potential)
    _, _, _ = plot_potential_1d_x(minimum, 0, (), ax=ax.twinx())

    fig.show()
    input()


def test_hamiltonian_large_resolution() -> None:
    """
    Test the generated hamiltonian when using a larger basis.

    We expect the result should be equal if we throw away the additional states.
    """
    resolution0 = (23, 23, 12)
    h0 = get_hamiltonian_hydrogen_sho(
        shape=(2 * resolution0[0], 2 * resolution0[1], 500),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution0,
    )

    h0_large = get_hamiltonian_hydrogen_sho(
        shape=(10 * resolution0[0], 10 * resolution0[1], 500),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution0,
    )
    np.testing.assert_array_almost_equal(h0_large["array"], h0["array"])
    # Max absolute difference: 6.531858e-24
    # Max relative difference: 119.78769123
    # Max value 3.04e-18
    np.testing.assert_array_equal(h0_large["array"], h0["array"])
    resolution1 = (25, 25, 16)
    h1 = get_hamiltonian_hydrogen_sho(
        shape=(2 * resolution1[0], 2 * resolution1[1], 500),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution1,
    )
    expected = pad_ft_points(
        h1["array"].reshape(*BasisUtil(h1["basis"]).shape, *BasisUtil(h1["dual_basis"]).shape),  # type: ignore[arg-type]
        (resolution0[0], resolution0[0], resolution0[0], resolution0[0]),
        axes=(0, 1, 3, 4),
    )[:, :, : resolution0[2], :, :, : resolution0[2]]
    np.testing.assert_array_almost_equal(
        expected,
        h0["array"].reshape(
            *BasisUtil(h0["basis"]).shape, *BasisUtil(h0["dual_basis"]).shape
        ),
    )
    # Max absolute difference: 8.18056993e-25
    # Max relative difference: 53.43269578
    np.testing.assert_array_equal(
        expected,
        h0["array"].reshape(
            *BasisUtil(h0["basis"]).shape, *BasisUtil(h0["dual_basis"]).shape
        ),
    )


def test_hamiltonian_very_large_resolution() -> None:
    """
    Test the generated hamiltonian when using a larger basis.

    We expect the result should be equal if we throw away the additional states.
    """
    resolution = (23, 23, 12)
    h0 = get_hamiltonian_hydrogen_sho(
        shape=(2 * resolution[0], 2 * resolution[1], 250),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution,
    )

    h1 = get_hamiltonian_hydrogen_sho(
        shape=(200, 200, 250),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution,
    )
    np.testing.assert_array_almost_equal(h1["array"], h0["array"])
    # Max absolute difference: 6.531858e-24
    # Max relative difference: 119.78769123
    # Max value 3.04e-18

    h2 = get_hamiltonian_hydrogen_sho(
        shape=(100, 100, 250),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution,
    )
    np.testing.assert_array_almost_equal(h1["array"], h2["array"])
    # Max absolute difference: 1.83756526e-24
    # Max relative difference: 13.57765139

    h3 = get_hamiltonian_hydrogen_sho(
        shape=(250, 250, 250),
        bloch_fraction=np.array([0, 0, 0]),
        resolution=resolution,
    )
    np.testing.assert_array_almost_equal(h1["array"], h3["array"])
    np.testing.assert_array_equal(h1["array"], h3["array"])
    # Max absolute difference: 3.62036857e-25
    # Max relative difference: 3.32098834
    # This is comparable to the size of the bandwidth (~1E-25)
    # And we would hope the errors cancel/ are not around the
    # minimum of energy in some way
