from __future__ import annotations

import itertools
from functools import cache
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Unpack

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.util import (
    AxisWithLengthBasisUtil,
    BasisUtil,
)
from surface_potential_analysis.dynamics.hermitian_gamma_integral import (
    calculate_hermitian_gamma_occupation_integral,
    calculate_hermitian_gamma_potential_integral,
)
from surface_potential_analysis.dynamics.lindbladian import (
    NonHermitianGamma,
    calculate_jump_operators,
    solve_master_equation,
)
from surface_potential_analysis.overlap.conversion import (
    convert_overlap_to_momentum_basis,
)
from surface_potential_analysis.overlap.interpolation import (
    get_angle_averaged_diagonal_overlap_function,
    get_overlap_momentum_interpolator_flat,
)
from surface_potential_analysis.overlap.plot import (
    plot_overlap_2d_k,
    plot_overlap_2d_x,
    plot_overlap_along_k_diagonal,
    plot_overlap_k0k1,
)
from surface_potential_analysis.util.constants import FERMI_WAVEVECTOR
from surface_potential_analysis.util.decorators import npy_cached

from .s4_wavepacket import (
    get_hydrogen_energy_difference,
    get_wavepacket_hydrogen,
)
from .s5_overlap import get_overlap_hydrogen
from .surface_data import get_data_path, save_figure

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from surface_potential_analysis._types import SingleIndexLike3d
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalMomentumBasis3d,
        FundamentalPositionBasis3d,
    )
    from surface_potential_analysis.dynamics.lindbladian import (
        NonHermitianGammaCoefficientMatrix,
    )
    from surface_potential_analysis.overlap.overlap import (
        FundamentalMomentumOverlap,
        Overlap3d,
    )

    _L0Inv = TypeVar("_L0Inv", bound=int)
    _L1Inv = TypeVar("_L1Inv", bound=int)
    _L2Inv = TypeVar("_L2Inv", bound=int)
    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _B0Inv = TypeVar("_B0Inv", bound=Basis3d[Any, Any, Any])


def get_max_point(
    overlap: FundamentalMomentumOverlap[_L0Inv, _L1Inv, _L2Inv]
) -> tuple[int, int, int]:
    points = overlap["vector"]
    (ik0, ik1, inz) = np.unravel_index(np.argmax(np.abs(points)), shape=points.shape)
    return (int(ik0), int(ik1), int(inz))


def make_overlap_real_at(
    overlap: Overlap3d[_B0Inv],
    idx: SingleIndexLike3d | None = None,
) -> Overlap3d[_B0Inv]:
    """
    Shift the phase of the overlap transform such that is it real at point.

    This is equivalent to shifting the origin of the real space grid we use when
    calculating the overlap.

    Parameters
    ----------
    overlap : OverlapMomentum
    point : tuple[int, int, int] | None, optional
        Point to make real, by default the maximum of the overlap

    Returns
    -------
    OverlapMomentum
        A new overlap, which is real at the given point
    """
    util = AxisWithLengthBasisUtil(overlap["basis"])
    idx = int(np.argmax(np.abs(overlap["vector"]))) if idx is None else idx
    idx = util.get_flat_index(idx) if isinstance(idx, tuple) else idx

    new_points = overlap["vector"] * np.exp(-1j * np.angle(overlap["vector"][idx]))
    return {"basis": overlap["basis"], "vector": new_points}


def calculate_average_overlap_nickel(
    i: int, j: int, *, r_samples: int = 50, theta_samples: int = 50
) -> np.float_:
    overlap = get_overlap_hydrogen(i, j)
    interpolator = get_overlap_momentum_interpolator_flat(overlap)

    radius = np.linspace(0, 1.77 * 10 ** (10), r_samples)
    theta = np.linspace(0, 2 * np.pi, theta_samples)
    averages = []
    for r in radius:
        k_points = r * np.array([np.cos(theta), np.sin(theta)])
        interpolated_vector = interpolator(k_points)  # type: ignore[var-annotated]
        averages.append(np.average(np.abs(interpolated_vector) ** 2))

    return np.average(averages, weights=radius)


def print_averaged_overlap_nickel() -> None:
    print(calculate_average_overlap_nickel(0, 1))  # noqa: T201
    print(calculate_average_overlap_nickel(0, 2))  # noqa: T201
    print(calculate_average_overlap_nickel(0, 3))  # noqa: T201
    print(calculate_average_overlap_nickel(1, 5))  # noqa: T201


def plot_angle_averaged_overlap() -> None:
    overlap = get_overlap_hydrogen(4, 2)
    abs_q = np.linspace(0, 5 * FERMI_WAVEVECTOR["NICKEL"], num=25)

    fig, ax = plt.subplots()

    util = BasisUtil(get_wavepacket_hydrogen(0)["basis"])
    interpolator = get_overlap_momentum_interpolator_flat(
        overlap, np.prod(util.shape[:2])
    )
    average_overlap = get_angle_averaged_diagonal_overlap_function(interpolator, abs_q)
    ax.plot(abs_q, average_overlap)

    interpolator = get_overlap_momentum_interpolator_flat(overlap, None)
    average_overlap = get_angle_averaged_diagonal_overlap_function(interpolator, abs_q)
    ax.plot(abs_q, average_overlap)
    fig.show()
    input()


def plot_overlap_momentum_interpolation() -> None:
    overlap = get_overlap_hydrogen(0, 1)

    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="abs")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    fig.show()

    k0_points = np.linspace(-2e11, 2e11, 1000)
    k_points = np.array([k0_points, k0_points])

    interpolated_vector = get_overlap_momentum_interpolator_flat(overlap)(k_points)  # type: ignore[var-annotated]
    fig, ax = plt.subplots()
    ax.plot(k0_points, np.abs(interpolated_vector))
    ax.plot(k0_points, np.real(interpolated_vector))
    ax.plot(k0_points, np.imag(interpolated_vector))
    fig.show()

    r = 1.77 * 10 ** (10)
    theta = np.linspace(0, 2 * np.pi, 100)
    k_points = r * np.array([np.cos(theta), np.sin(theta)])
    interpolated_vector = get_overlap_momentum_interpolator_flat(overlap)(k_points)  # type: ignore[var-annotated]
    fig, ax = plt.subplots()
    ax.plot(theta, np.abs(interpolated_vector))
    ax.plot(theta, np.real(interpolated_vector))
    ax.plot(theta, np.imag(interpolated_vector))

    fig.show()
    input()


def plot_fcc_hcp_overlap_momentum() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="abs")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="real")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_real_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (1, 0), (0,), measure="imag")
    ax.set_title(
        "Plot of the overlap in momentum for ikz=0\n"
        "showing oscillation in the direction corresponding to\n"
        "a vector spanning the fcc and hcp sites"
    )
    save_figure(fig, "2d_overlap_transform_imag_kx_ky.png")
    fig.show()

    x1_max = np.unravel_index(
        np.argmax(overlap["vector"]),
        AxisWithLengthBasisUtil(overlap_momentum["basis"]).shape,
    )[1]
    fig, ax, _ = plot_overlap_2d_k(overlap_momentum, (0, 2), (x1_max,))
    ax.set_title(
        "Plot of the overlap in momentum for ikx1=0\n"
        "A very sharp peak in the kz direction"
    )
    ax.set_ylim(-4e11, 4e11)
    save_figure(fig, "2d_overlap_fraction_kx1_kz.png")
    fig.show()
    input()


def plot_fcc_hcp_overlap_momentum_along_diagonal() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)

    fig, ax = plt.subplots()
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, 2, measure="abs", ax=ax)
    ln.set_label("abs")
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, 2, measure="real", ax=ax)
    ln.set_label("real")
    _, _, ln = plot_overlap_along_k_diagonal(overlap_momentum, 2, measure="imag", ax=ax)
    ln.set_label("imag")

    ax.legend()
    ax.set_title(
        "Plot of the wavefunction along the diagonal,\nshowing an oscillation in the overlap"
    )

    save_figure(fig, "diagonal_1d_overlap_fraction.png")
    fig.show()
    input()


def plot_fcc_hcp_overlap() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    x2_max = np.unravel_index(
        np.argmax(overlap["vector"]), AxisWithLengthBasisUtil(overlap["basis"]).shape
    )[2]

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (x2_max,), measure="abs")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_kx_ky.png")
    fig.show()

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (x2_max,), measure="real")
    ax.set_title(
        "Plot of the overlap summed over z\n"
        "showing the FCC and HCP asymmetry\n"
        "in a small region in the center of the figure"
    )
    save_figure(fig, "2d_overlap_real_kx_ky.png")
    fig.show()
    input()


def plot_fcc_hcp_overlap_offset() -> None:
    overlap = get_overlap_hydrogen(0, 1, (1, 0))
    x2_max = np.unravel_index(
        np.argmax(overlap["vector"]), AxisWithLengthBasisUtil(overlap["basis"]).shape
    )[2]

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (x2_max,), measure="abs")
    fig.show()

    overlap = get_overlap_hydrogen(0, 1, (0, 0), (-1, 0))
    x2_max = np.unravel_index(
        np.argmax(overlap["vector"]), AxisWithLengthBasisUtil(overlap["basis"]).shape
    )[2]

    fig, ax, _ = plot_overlap_2d_x(overlap, (0, 1), (x2_max,), measure="abs")
    fig.show()
    input()


def get_angle_averaged_overlap(
    interpolator_1: Callable[
        [np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]],
        np.ndarray[tuple[int], np.dtype[np.complex_]],
    ],
    interpolator_2: Callable[
        [np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]],
        np.ndarray[tuple[int], np.dtype[np.complex_]],
    ],
    abs_q: np.ndarray[tuple[int], np.dtype[np.float_]],
    *,
    theta_samples: int = 50,
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    theta = np.linspace(0, 2 * np.pi, theta_samples)
    averages = []
    for q in abs_q:
        k_points = q * np.array([np.cos(theta), np.sin(theta)])
        interpolated_1 = interpolator_1(k_points)  # type: ignore[var-annotated]
        interpolated_2 = interpolator_2(k_points)  # type: ignore[var-annotated]
        averages.append(np.average(interpolated_1 * np.conj(interpolated_2)))

    return np.array(averages, dtype=complex)  # type: ignore[no-any-return]


def plot_angle_averaged_overlap_momentum() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    interpolator = get_overlap_momentum_interpolator_flat(overlap)
    radius = np.linspace(0, 1.5e11, 100)
    averaged = get_angle_averaged_overlap(interpolator, interpolator, radius)

    fig, ax = plt.subplots()
    ax.plot(radius, averaged)
    fig.show()
    input()


def shift_offset_0_to_origin(
    offset: np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]]
) -> np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]]:
    return offset - offset[0][np.newaxis, :]  # type: ignore[no-any-return]


def shift_relative_offset(
    offset: np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]]
) -> tuple[
    np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]],
    np.ndarray[tuple[Literal[2]], np.dtype[np.int_]],
]:
    """Shift the second pair of offsets to the origin, and return the distance required."""
    displacement = offset[2]
    offset[2:4, :] -= displacement[np.newaxis, :]
    return offset, displacement


def is_offset_irrelevant(
    offset: np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]]
) -> bool:
    too_large_i = not (
        -2 < offset[1, 0] - offset[0, 0] < 2  # noqa: PLR2004
        and -2 < offset[1, 1] - offset[0, 1] < 2  # noqa: PLR2004
    )
    too_large_j = not (
        -2 < offset[3, 0] - offset[2, 0] < 2  # noqa: PLR2004
        and -2 < offset[3, 1] - offset[2, 1] < 2  # noqa: PLR2004
    )
    return too_large_i or too_large_j


@cache
def get_overlap_interpolator_cached(
    i: int, j: int, offset: tuple[int, int]
) -> Callable[
    [np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]],
    np.ndarray[_S0Inv, np.dtype[np.complex_]],
]:
    overlap = get_overlap_hydrogen(i, j, (0, 0), (offset[0], offset[1]))
    return get_overlap_momentum_interpolator_flat(overlap)


def get_gamma_1(
    idx: np.ndarray[tuple[Literal[4]], np.dtype[np.int_]],
    offset: np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]],
) -> np.complex_:
    offset = shift_offset_0_to_origin(offset)
    offset, displacement = shift_relative_offset(offset)

    util = AxisWithLengthBasisUtil(get_wavepacket_hydrogen(0)["basis"])
    displacement_r = np.tensordot(displacement, util.delta_x[0:2], axes=(0, 0))[:2]
    interpolator_0 = get_overlap_interpolator_cached(
        idx.item(0), idx.item(1), tuple(offset[1])
    )
    interpolator_1 = get_overlap_interpolator_cached(
        idx.item(2), idx.item(3), tuple(offset[3])
    )

    def interpolator_1_shifted(
        q: np.ndarray[tuple[Literal[2], Any], np.dtype[np.float_]]
    ) -> np.ndarray[Any, np.dtype[np.complex_]]:
        return interpolator_1(q) * np.exp(  # type: ignore[no-any-return]
            1j * np.tensordot(q, displacement_r, axes=(0, 0))
        )

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return get_angle_averaged_overlap(
            interpolator_0, interpolator_1_shifted, q.ravel()
        ).reshape(q.shape)

    return calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR["NICKEL"], overlap_function
    )


def get_gamma_2(i: int, j: int, temperature: float) -> float:
    if i == j:
        return calculate_hermitian_gamma_occupation_integral(
            0, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
        )
    omega = float(get_hydrogen_energy_difference(0, 1))
    if i == 0 and j == 1:
        return calculate_hermitian_gamma_occupation_integral(
            -omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
        )
    if i == 1 and j == 0:
        return calculate_hermitian_gamma_occupation_integral(
            omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
        )
    raise NotImplementedError


def calculate_gamma(  # noqa: PLR0913
    i: int,
    i1: int,
    k: int,
    k1: int,
    offset: np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.int_]],
    temperature: float,
) -> float:
    non_zero = (i == k and i1 == k1) or (i == k1 and k == i1)
    if not non_zero or is_offset_irrelevant(offset):
        return 0.0

    gamma_1 = get_gamma_1(np.array([i, i1, k, k1]), offset)
    gamma_2 = get_gamma_2(i, i1, temperature)
    return float(gamma_1 * gamma_2)


@npy_cached(get_data_path("GammaFirstAtempt.npy"))
def build_gamma(shape: tuple[int, int], temperature: float) -> NonHermitianGamma:
    n_sites = np.product(shape)

    gamma = np.zeros(
        (2 * n_sites, 2 * n_sites, 2 * n_sites, 2 * n_sites), dtype=np.complex_
    )

    for i in range(0, 1):
        for i1 in range(0, 1):
            for k in range(0, 1):
                for k1 in range(0, 1):
                    for site_0 in range(n_sites):
                        for hop_1 in range(9):
                            for hop_2 in range(n_sites):
                                for hop_3 in range(9):
                                    idx_0 = np.ravel_multi_index(
                                        (i, site_0), (2, n_sites)
                                    )
                                    hop_1_stacked = np.unravel_index(
                                        hop_1, (3, 3)
                                    ) - np.array([1, 1])
                                    site_0_stacked = np.unravel_index(site_0, shape)
                                    site_1_stacked = site_0_stacked + hop_1_stacked
                                    idx_1 = np.ravel_multi_index(
                                        (i1, *site_1_stacked), (2, *shape), mode="wrap"
                                    )

                                    hop_2_stacked = np.unravel_index(
                                        hop_2, (3, 3)
                                    ) - np.array([shape[0] // 2, shape[1] // 2])
                                    site_2_stacked = site_0_stacked + hop_2_stacked
                                    idx_2 = np.ravel_multi_index(
                                        (k, *site_2_stacked), (2, *shape), mode="wrap"
                                    )
                                    hop_3_stacked = np.unravel_index(
                                        hop_3, (3, 3)
                                    ) - np.array([1, 1])
                                    site_3_stacked = site_2_stacked + hop_3_stacked
                                    idx_3 = np.ravel_multi_index(
                                        (k1, *site_3_stacked), (2, *shape), mode="wrap"
                                    )

                                    offset = np.array(
                                        [
                                            site_0_stacked,
                                            site_1_stacked,
                                            site_2_stacked,
                                            site_3_stacked,
                                        ]
                                    )
                                    gamma[idx_0, idx_1, idx_2, idx_3] = calculate_gamma(
                                        i, i1, k, k1, offset, temperature
                                    )
    n_gamma = np.square(2 * n_sites)
    return {"array": gamma.reshape(n_gamma, n_gamma)}


def test_interpolation_shifted() -> None:
    overlap_0_1_next = get_overlap_hydrogen(0, 1, (1, 0))
    interpolator_0_1_next = get_overlap_momentum_interpolator_flat(overlap_0_1_next)

    overlap_0_1_previous = get_overlap_hydrogen(0, 1, (0, 0), (-1, 0))
    interpolator_0_1_previous = get_overlap_momentum_interpolator_flat(
        overlap_0_1_previous
    )
    q_points = np.linspace(0, FERMI_WAVEVECTOR["NICKEL"], 1000)
    q = np.array([np.zeros_like(q_points), q_points])
    np.testing.assert_almost_equal(
        interpolator_0_1_next(q), interpolator_0_1_previous(q)
    )

    basis = get_wavepacket_hydrogen(0)["basis"]
    q = np.array([q_points, np.zeros_like(q_points)])
    np.testing.assert_almost_equal(
        interpolator_0_1_next(q),
        interpolator_0_1_previous(q)
        * np.exp(1j * np.tensordot(q, basis[0].delta_x[0:2], axes=(0, 0))),
    )


def calculate_potential_integral() -> None:
    # !overlap_0_0 = get_overlap(0, 0)
    # !interpolator_0_0 = get_overlap_momentum_interpolator_flat(overlap_0_0)
    # !
    # !def overlap_function(
    # !    r: np.ndarray[_S0Inv, np.dtype[np.float_]]
    # !) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    # !    return get_angle_averaged_overlap_nickel(
    # !        interpolator_0_0, interpolator_0_0, r.ravel()
    # !    ).reshape(r.shape)
    # !
    # !integral = calculate_hermitian_gamma_potential_integral(
    # !    FERMI_WAVEVECTOR["NICKEL"], overlap_function
    # !)
    # !print("(0, 0)", integral)
    # !
    # !overlap_1_1 = get_overlap(1, 1)
    # !interpolator_1_1 = get_overlap_momentum_interpolator_flat(overlap_1_1)
    # !
    # !def overlap_function(
    # !    q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    # !) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    # !    return get_angle_averaged_overlap_nickel(
    # !        interpolator_1_1, interpolator_1_1, q.ravel()
    # !    ).reshape(q.shape)
    # !
    # !integral = calculate_hermitian_gamma_potential_integral(
    # !    FERMI_WAVEVECTOR["NICKEL"], overlap_function
    # !)
    # !print("(1, 1)", integral)

    overlap_0_1 = get_overlap_hydrogen(0, 1)
    interpolator_0_1 = get_overlap_momentum_interpolator_flat(overlap_0_1)
    # !
    # !def overlap_function(
    # !    q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    # !) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    # !    return get_angle_averaged_overlap_nickel(
    # !        interpolator_0_1, interpolator_0_1, q.ravel()
    # !    ).reshape(q.shape)
    # !
    # !integral = calculate_hermitian_gamma_potential_integral(
    # !    FERMI_WAVEVECTOR["NICKEL"], overlap_function
    # !)
    # !print("(0, 1)", integral)

    overlap_0_1_next = get_overlap_hydrogen(0, 1, (1, 0))
    interpolator_0_1_next = get_overlap_momentum_interpolator_flat(overlap_0_1_next)

    def overlap_function(
        q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    ) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
        return get_angle_averaged_overlap(
            interpolator_0_1, interpolator_0_1_next, q.ravel()
        ).reshape(q.shape)

    integral = calculate_hermitian_gamma_potential_integral(
        FERMI_WAVEVECTOR["NICKEL"], overlap_function
    )
    print("(0, 1) (next 0, 1)", integral)  # noqa: T201

    overlap_0_1_previous = get_overlap_hydrogen(0, 1, (0, 0), (0, -1))
    get_overlap_momentum_interpolator_flat(overlap_0_1_previous)

    # !def overlap_function(
    # !    q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    # !) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    # !    return get_angle_averaged_overlap_nickel(
    # !        interpolator_0_1, interpolator_0_1_previous, q.ravel()
    # !    ).reshape(q.shape)
    # !
    # !integral = calculate_hermitian_gamma_potential_integral(
    # !    FERMI_WAVEVECTOR["NICKEL"], overlap_function
    # !)
    # !print("(0, 1) (0, previous 1)", integral)

    # !overlap_0_1_next2 = get_overlap(0, 1, (2, 0))
    # !interpolator_0_1_next2 = get_overlap_momentum_interpolator_flat(overlap_0_1_next2)
    # !
    # !def overlap_function(
    # !    q: np.ndarray[_S0Inv, np.dtype[np.float_]]
    # !) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    # !    return get_angle_averaged_overlap_nickel(
    # !        interpolator_0_1, interpolator_0_1_next2, q.ravel()
    # !    ).reshape(q.shape)
    # !
    # !integral = calculate_hermitian_gamma_potential_integral(
    # !    FERMI_WAVEVECTOR["NICKEL"], overlap_function
    # !)
    # !print("(0, 1) (1, next next 0)", integral)

    # ! (0, 0) (3319708.634328286+0j)
    # ! (1, 1) (3317746.123271401+0j)
    # ! (0, 1) (26.930494462707472+0j)
    # ! (0, 1) (1, next next 0) (9.586706390306949+1.3143994837175386j)
    # ! (0, 1) (1, next next 0) (-0.0005268934170412702-0.00011696603364330633j)


def plot_temperature_dependent_integral() -> None:
    temperatures = np.linspace(50, 200, 50)
    vals = [
        calculate_hermitian_gamma_occupation_integral(
            0, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * t
        )
        for t in temperatures
    ]
    fig, ax = plt.subplots()
    ax.plot(1 / temperatures, vals)

    fig.show()
    input()


def calculate_max_overlap(
    overlap: Overlap3d[FundamentalPositionBasis3d[_L0Inv, _L1Inv, _L2Inv]],
) -> tuple[complex, np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]]:
    points = overlap["vector"]
    util = AxisWithLengthBasisUtil(overlap["basis"])
    arg_max = np.argmax(np.abs(points))
    x_point = util.fundamental_x_points[:, arg_max]

    return points[arg_max], x_point


def calculate_max_overlap_momentum(
    overlap: Overlap3d[FundamentalMomentumBasis3d[_L0Inv, _L1Inv, _L2Inv]],
) -> tuple[complex, np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]]:
    points = overlap["vector"]
    util = AxisWithLengthBasisUtil(overlap["basis"])
    arg_max = np.argmax(np.abs(points))
    k_point = util.fundamental_k_points[:, arg_max]

    return points[arg_max], k_point


def print_max_overlap_momentum() -> None:
    overlap = get_overlap_hydrogen(0, 1)
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    print(overlap_momentum["vector"][0])  # noqa: T201
    print(calculate_max_overlap_momentum(overlap_momentum))  # noqa: T201

    overlap = get_overlap_hydrogen(0, 0, (0, 0))
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    print(overlap_momentum["vector"][0])  # noqa: T201
    print(calculate_max_overlap_momentum(overlap_momentum))  # noqa: T201

    overlap = get_overlap_hydrogen(1, 1, (0, 0))
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    print(overlap_momentum["vector"][0])  # noqa: T201
    print(calculate_max_overlap_momentum(overlap_momentum))  # noqa: T201

    overlap = get_overlap_hydrogen(0, 0, (1, 0))
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    print(overlap_momentum["vector"][0])  # noqa: T201
    print(calculate_max_overlap_momentum(overlap_momentum))  # noqa: T201

    overlap = get_overlap_hydrogen(1, 1, (1, 0))
    overlap_momentum = convert_overlap_to_momentum_basis(overlap)
    print(overlap_momentum["vector"][0])  # noqa: T201
    print(calculate_max_overlap_momentum(overlap_momentum))  # noqa: T201


def print_max_and_min_overlap() -> None:
    for i in range(6):
        for j in range(i + 1, 6):
            overlap = get_overlap_hydrogen(i, j)
            overlap_momentum = convert_overlap_to_momentum_basis(overlap)
            print(f"i={i}, j={j}")  # noqa: T201
            max_overlap = calculate_max_overlap_momentum(overlap_momentum)
            print(max_overlap)  # noqa: T201
            print(np.abs(max_overlap[0]), np.linalg.norm(max_overlap[1]))  # noqa: T201
            k0_momentum = overlap_momentum["vector"][0]
            print(k0_momentum)  # noqa: T201
            print(np.abs(k0_momentum))  # noqa: T201


def plot_all_abs_overlap_k() -> None:
    for i in range(6):
        for j in range(i + 1, 6):
            overlap = get_overlap_hydrogen(i, j)
            overlap_transform = convert_overlap_to_momentum_basis(overlap)

            fig, ax, _ = plot_overlap_k0k1(overlap_transform, 0, measure="abs")
            ax.set_title(f"Plot of the overlap transform for k2=0\nfor i={i} and j={j}")
            save_figure(fig, f"overlap/abs_overlap_k0k1_plot_{i}_{j}.png")
            fig.show()
    input()


def load_average_band_energies(
    n_bands: _L0Inv,
) -> np.ndarray[tuple[_L0Inv], np.dtype[np.float_]]:
    energies = np.zeros((n_bands,))
    for band in range(n_bands):
        wavepacket = get_wavepacket_hydrogen(band)
        energies[band] = wavepacket["eigenvalues"][0, 0]
    return energies  # type: ignore[no-any-return]


def _build_incoherent_matrix_cache(n_bands: _L0Inv, _temperature: float = 150) -> Path:
    return get_data_path(f"incoherent_matrix_{n_bands}_bands_{_temperature}k.npy")


@npy_cached(_build_incoherent_matrix_cache)
def build_incoherent_matrix(
    n_bands: _L0Inv, _temperature: float = 150
) -> Any:  # noqa: ANN401
    # The coefficients np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    # represent the total rate R[i,j,dx] from i to j with an offset of dx at the location i.
    load_average_band_energies(n_bands)
    1.77 * 10 ** (10)
    out = np.zeros((n_bands, n_bands, 9))
    for i, j, dx0, dx1 in itertools.product(
        range(n_bands), range(n_bands), range(-1, 2), range(-1, 2)
    ):
        offset = (dx0, dx1)
        print(f"i={i}, j={j} offset={offset}")  # noqa: T201
        overlap = get_overlap_hydrogen(i, j, offset)
        overlap_momentum = convert_overlap_to_momentum_basis(overlap)

        max_overlap, _ = calculate_max_overlap_momentum(overlap_momentum)
        # ! prefactor = np.exp(-(energies[j] - energies[i]) / (Boltzmann * 150))
        # ! energy_jump = energies[j] - energies[i]
        # ! prefactor = calculate_electron_integral(fermi_k, energy_jump, temperature)
        prefactor = 1

        dx = np.ravel_multi_index(offset, (3, 3), mode="wrap")
        out[i, j, dx] = np.abs(max_overlap) ** 2 * prefactor

    return out  # type: ignore [no-any-return]


def build_incoherent_matrix_fcc_hcp() -> (
    np.ndarray[tuple[Literal[2], Literal[2], Literal[9]], np.dtype[np.float_]]
):
    # The coefficients np.ndarray[tuple[_L0Inv, _L0Inv, Literal[9]], np.dtype[np.float_]]
    # represent the total rate R[i,j,dx] from i to j with an offset of dx at the location i.
    energies = load_average_band_energies(2)
    out = np.zeros((2, 2, 9))
    rate = 2.56410914e-05
    exponential = np.exp(-(energies[0] - energies[1]) / (Boltzmann * 150))
    print(exponential)  # noqa: T201
    out[0, 1, 0] = rate * exponential
    out[0, 1, np.ravel_multi_index((-1, 0), (3, 3), mode="wrap")] = rate * exponential
    out[0, 1, np.ravel_multi_index((0, -1), (3, 3), mode="wrap")] = rate * exponential
    out[1, 0, 0] = rate / exponential
    out[1, 0, np.ravel_multi_index((1, 0), (3, 3), mode="wrap")] = rate / exponential
    out[1, 0, np.ravel_multi_index((0, 1), (3, 3), mode="wrap")] = rate / exponential
    return out  # type: ignore [no-any-return]


def build_gamma_coefficient_matrix_fcc_hcp(
    temperature: float,
) -> NonHermitianGammaCoefficientMatrix[Literal[2]]:
    out = np.zeros((2, 2, 9))
    constant_rate = 26.93
    omega = float(get_hydrogen_energy_difference(0, 1))

    fast_rate = constant_rate * calculate_hermitian_gamma_occupation_integral(
        omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
    )
    slow_rate = constant_rate * calculate_hermitian_gamma_occupation_integral(
        omega, FERMI_WAVEVECTOR["NICKEL"], Boltzmann * temperature
    )

    out[0, 1, 0] = fast_rate
    out[0, 1, np.ravel_multi_index((-1, 0), (3, 3), mode="wrap")] = fast_rate
    out[0, 1, np.ravel_multi_index((0, -1), (3, 3), mode="wrap")] = fast_rate
    out[1, 0, 0] = slow_rate
    out[1, 0, np.ravel_multi_index((1, 0), (3, 3), mode="wrap")] = slow_rate
    out[1, 0, np.ravel_multi_index((0, 1), (3, 3), mode="wrap")] = slow_rate
    return {"array": out}


def solve_master_equation_nickel() -> None:
    # ! coefficient_matrix = build_gamma_coefficient_matrix_fcc_hcp(150)
    # ! gamma = calculate_gamma_two_state((3, 3), coefficient_matrix)
    gamma = build_gamma((3, 3), 150)
    print(gamma["array"].shape)  # noqa: T201
    jump_operators = calculate_jump_operators(gamma)
    _solution = solve_master_equation(jump_operators)
    print(_solution)  # noqa: T201
