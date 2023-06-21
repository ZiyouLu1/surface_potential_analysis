from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.build import (
    position_basis_3d_from_parent,
    position_basis_3d_from_shape,
)
from surface_potential_analysis.basis.plot import (
    plot_bragg_points_projected_2d,
    plot_brillouin_zone_points_projected_2d,
    plot_fundamental_k_in_plane_projected_2d,
    plot_fundamental_x_at_index_projected_2d,
    plot_fundamental_x_in_plane_projected_2d,
)
from surface_potential_analysis.basis.util import Basis3dUtil

from .s1_potential import get_interpolated_potential

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from surface_potential_analysis._types import (
        SingleStackedIndexLike3d,
    )
    from surface_potential_analysis.basis.basis import (
        Basis3d,
        FundamentalPositionBasis3d,
    )

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
    _B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])


def load_nickel_single_point_basis() -> (
    FundamentalPositionBasis3d[Literal[1], Literal[1], Literal[1]]
):
    return get_interpolated_potential((1, 1, 1))["basis"]  # type: ignore[return-value]


def is_coordinate_inside_1st_bz(basis: Basis3d, coordinate: SingleStackedIndexLike3d):
    util = Basis3dUtil(basis)
    (a0, a1, a2) = coordinate
    inside = True
    for n0, n1, n2 in [
        (util.n0 / 2, 0, 0),
        (0, util.n1 / 2, 0),
        (0, 0, util.n2 / 2),
        (util.n0 / 2, util.n1 / 2, 0),
        (util.n0 / 2, 0, util.n2 / 2),
        (0, util.n1 / 2, util.n2 / 2),
        (util.n0 / 2, util.n1 / 2, util.n2 / 2),
    ]:
        # given a normal n, vector a in the plane and a point p
        # d = n.(a-p)is the distance in the direction of n of the vector a-p
        if n0 * (a0 - n0) + n1 * (a1 - n1) + n2 * (a2 - n2) > 0:
            inside = False
    return inside


def plot_bragg_planes(
    basis: _B3d0Inv, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = Basis3dUtil(basis)

    _site_1 = (util.n0, 0)  # type: ignore[misc]
    _site_2 = (0, util.n1)  # type: ignore[misc]
    _site_3 = (util.n0, util.n1)  # type: ignore[misc]
    _vec_0 = np.tensordot(
        [util.fundamental_dk0, util.fundamental_dk1], (0, 0), axes=(0, 0)
    )
    _vec_1 = np.tensordot(
        [util.fundamental_dk0, util.fundamental_dk1], (util.n0 / 2, 0), axes=(0, 0)  # type: ignore[misc]
    )
    _vec_2 = np.tensordot(
        [util.fundamental_dk0, util.fundamental_dk1], (0, util.n1 / 2), axes=(0, 0)  # type: ignore[misc]
    )
    _vec_3 = np.tensordot(
        [util.fundamental_dk0, util.fundamental_dk1],
        (util.n0 / 2, util.n1 / 2),  # type: ignore[misc]
        axes=(0, 0),
    )
    (a0, a1) = (1, 2)
    _is_inside = True
    for n0, n1 in [(util.n0 / 2, 0), (0, util.n1 / 2), (util.n0 / 2, util.n1 / 2)]:  # type: ignore[misc]
        # given a normal n, vector a in the plane and a point p
        # d = n.(a-p)is the distance in the direction of n of the vector a-p
        if n0 * (a0 - n0) + n1 * (a1 - n1) > 0:
            _is_inside = False
    (line,) = ax.plot(_vec_0[0], _vec_0[1])
    line.set_marker("x")
    (line,) = ax.plot(_vec_1[0], _vec_1[1])
    line.set_marker("x")
    (line,) = ax.plot(_vec_2[0], _vec_2[1])
    line.set_marker("x")
    (line,) = ax.plot(_vec_3[0], _vec_3[1])
    line.set_marker("x")
    return fig, ax


def plot_brillouin_zone_points() -> None:
    parent = load_nickel_single_point_basis()
    basis = position_basis_3d_from_parent(parent, (6, 6, 100))
    fig, _, _ = plot_fundamental_x_at_index_projected_2d(basis, np.arange(36), 2)
    fig.show()

    fig, _, _ = plot_fundamental_x_in_plane_projected_2d(basis, (0, 1), (0,))
    fig.show()

    fig, ax, _ = plot_fundamental_k_in_plane_projected_2d(basis, (0, 1), (0,))
    plot_bragg_planes(basis, ax=ax)
    fig.show()

    fig, ax, _ = plot_brillouin_zone_points_projected_2d(basis, (0, 1))
    plot_bragg_points_projected_2d(basis, (0, 1), ax=ax)
    fig.show()

    basis = position_basis_3d_from_shape((6, 6, 6))
    fig, ax, _ = plot_brillouin_zone_points_projected_2d(basis, (0, 1))
    plot_bragg_points_projected_2d(basis, (0, 1), ax=ax)
    fig.show()
    input()
