from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from surface_potential_analysis.axis.axis import (
    FundamentalPositionBasis,
    FundamentalPositionBasis3d,
)
from surface_potential_analysis.axis.stacked_axis import StackedBasis, StackedBasisLike
from surface_potential_analysis.potential.point_potential import (
    PointPotential3d,
    load_point_potential_json,
)
from surface_potential_analysis.potential.potential import (
    Potential,
    UnevenPotential3dZBasis,
    interpolate_uneven_potential,
    normalize_potential,
    truncate_potential,
    undo_truncate_potential,
)

from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import UnevenPotential3d


def load_raw_data() -> PointPotential3d[Any]:
    path = get_data_path("raw_data.json")
    points = load_point_potential_json(path)
    max_point = np.max(points["points"])
    min_point = np.min(points["points"])
    points["points"][np.argmin(points["points"])] = 1.3 * max_point - 0.3 * min_point
    return points


def map_irreducible_points_into_unit_cell(
    irreducible_points: PointPotential3d[int],
) -> UnevenPotential3d[int, int, int]:
    z_points = np.sort(np.unique(irreducible_points["z_points"]))
    xy_points = np.unique(
        np.array([irreducible_points["x_points"], irreducible_points["y_points"]]).T,
        axis=0,
    )

    mapping_to_xy = {
        (8, -4): (11, 4),
        (6, -3): (9, 3),
        (7, -3): (10, 4),
        (4, -2): (6, 2),
        (5, -2): (7, 3),
        (6, -2): (8, 4),
        (7, -2): (9, 5),
        (2, -1): (3, 1),
        (3, -1): (4, 2),
        (4, -1): (5, 3),
        (5, -1): (6, 4),
        (6, -1): (7, 5),
        (0, +0): (0, 0),
        (1, +0): (1, 1),
        (2, +0): (2, 2),
        (3, +0): (3, 3),
        (4, +0): (4, 4),
        (5, +0): (5, 5),
        (6, +0): (6, 6),
        (1, +1): (0, 2),
        (2, +1): (1, 3),
        (3, +1): (2, 4),
        (4, +1): (3, 5),
        (5, +1): (4, 6),
        (2, +2): (0, 4),
        (3, +2): (1, 5),
        (4, +2): (2, 6),
        (5, +2): (3, 7),
        (3, +3): (0, 6),
        (4, +3): (1, 7),
        (4, +4): (0, 8),
    }

    # Should really be this way round (ie when we plot), and then reverse the y direction and swap x, y axis to get the usual indexing
    # ruff: noqa: ERA001
    # mapping = [
    #     [(+8,-4), (+7,-3), (+6,-2), (+5,-1), (+4,+0), (+3,+1), (+2,+2), (+3,+1), (+4,+0), (+5,-1), (+6,-2), (+7,-3), (+8,-4)],
    #     [(+7,-3), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+6,-3), (+7,-3)],
    #     [(+6,-2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+4,-2), ( _, _), (+6,-2)],
    #     [(+5,-1), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+2,-1), ( _, _), ( _, _), (+5,-1)],
    #     [(+4,+0), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+0,+0), ( _, _), ( _, _), ( _, _), (+4,+0)],
    #     [(+3,+1), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+1,+1), ( _, _), ( _, _), ( _, _), ( _, _), (+3,+1)],
    #     [(+2,+2), (+3,+2), (+4,+2), (+5,+2), (+4,+2), (+3,+2), (+2,+2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+2,+2)],
    #     [(+3,+1), (+3,+2), (+3,+3), (+4,+3), (+4,+3), (+3,+3), (+3,+2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+3,+1)],
    #     [(+4,+0), (+4,+1), (+4,+2), (+4,+3), (+4,+4), (+4,+3), (+4,+2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+4,+0)],
    #     [(+5,-1), (+5,+0), (+5,+1), (+5,+2), (+4,+3), (+4,+3), (+5,+2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+5,-1)],
    #     [(+6,-2), (+6,-1), (+6,+0), (+5,+1), (+4,+2), (+3,+3), (+4,+2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+6,-2)],
    #     [(+7,-3), (+7,-2), (+6,-1), (+5,+0), (+4,+1), (+3,+2), (+3,+2), ( _, _), ( _, _), ( _, _), ( _, _), ( _, _), (+7,-3)],
    #     [(+8,-4), (+7,-3), (+6,-2), (+5,-1), (+4,+0), (+3,+1), (+2,+2), (+3,+1), (+4,+0), (+5,-1), (+6,-2), (+7,-3), (+8,-4)],
    # ]

    # fmt: off
    mapping = [
        [(+6, -3), (+5, -2), (+4, -1), (+3, +0), (+2, +1), (+2, +1), (+3, +0), (+4, -1), (+5, -2), (+6, -3), (+7, -3), (+7, -3)],
        [(+5, -2), (+4, -2), (+3, -1), (+2, +0), (+1, +1), (+2, +0), (+3, -1), (+4, -2), (+5, -2), (+6, -2), (+7, -2), (+6, -2)],
        [(+4, -1), (+3, -1), (+2, -1), (+1, +0), (+1, +0), (+2, -1), (+3, -1), (+4, -1), (+5, -1), (+6, -1), (+6, -1), (+5, -1)],
        [(+3, +0), (+2, +0), (+1, +0), (+0, +0), (+1, +0), (+2, +0), (+3, +0), (+4, +0), (+5, +0), (+6, +0), (+5, +0), (+4, +0)],
        [(+2, +1), (+1, +1), (+1, +0), (+1, +0), (+1, +1), (+2, +1), (+3, +1), (+4, +1), (+5, +1), (+5, +1), (+4, +1), (+3, +1)],
        [(+2, +1), (+2, +0), (+2, -1), (+2, +0), (+2, +1), (+2, +2), (+3, +2), (+4, +2), (+5, +2), (+4, +2), (+3, +2), (+2, +2)],
        [(+3, +0), (+3, -1), (+3, -1), (+3, +0), (+3, +1), (+3, +2), (+3, +3), (+4, +3), (+4, +3), (+3, +3), (+3, +2), (+3, +1)],
        [(+4, -1), (+4, -2), (+4, -1), (+4, +0), (+4, +1), (+4, +2), (+4, +3), (+4, +4), (+4, +3), (+4, +2), (+4, +1), (+4, +0)],
        [(+5, -2), (+5, -2), (+5, -1), (+5, +0), (+5, +1), (+5, +2), (+4, +3), (+4, +3), (+5, +2), (+5, +1), (+5, +0), (+5, -1)],
        [(+6, -3), (+6, -2), (+6, -1), (+6, +0), (+5, +1), (+4, +2), (+3, +3), (+4, +2), (+5, +1), (+6, +0), (+6, -1), (+6, -2)],
        [(+7, -3), (+7, -2), (+6, -1), (+5, +0), (+4, +1), (+3, +2), (+3, +2), (+4, +1), (+5, +0), (+6, -1), (+7, -2), (+7, -3)],
        [(+7, -3), (+6, -2), (+5, -1), (+4, +0), (+3, +1), (+2, +2), (+3, +1), (+4, +0), (+5, -1), (+6, -2), (+7, -3), (+8, -4)],
    ]
    # fmt: on

    # So it matches up with what we have for Ni, make the HCP site in the lower half
    mapping = np.array(mapping).swapaxes(0, 1)[::-1, ::-1].tolist()

    mapping_in_xy = [[mapping_to_xy[(k[0], k[1])] for k in m] for m in mapping]

    z_points_in_xy: dict[tuple[int, int], list[float]] = {}

    for z in z_points:
        for x, y in xy_points:
            ix = np.argwhere(x == np.sort(np.unique(irreducible_points["x_points"])))[
                0
            ][0]
            iy = np.argwhere(y == np.sort(np.unique(irreducible_points["y_points"])))[
                0
            ][0]

            curr = z_points_in_xy.get((ix, iy), [])

            is_at_xyz = np.logical_and(
                np.logical_and(
                    np.array(irreducible_points["x_points"]) == x,
                    np.array(irreducible_points["y_points"]) == y,
                ),
                np.array(irreducible_points["z_points"]) == z,
            )
            if np.count_nonzero(is_at_xyz) != 1:
                raise AssertionError("More than one point found")  # noqa: TRY003, EM101
            curr.append(np.array(irreducible_points["points"])[is_at_xyz][0])
            z_points_in_xy[(ix, iy)] = curr

    final_grid = np.array(
        [[z_points_in_xy[xy_coord] for xy_coord in m] for m in mapping_in_xy]
    )

    x_width = np.max(irreducible_points["x_points"]) - np.min(  # type: ignore[operator]
        irreducible_points["x_points"]
    )
    y_height = np.max(irreducible_points["y_points"]) - np.min(  # type: ignore[operator]
        irreducible_points["y_points"]
    )
    diagonal_length = np.sqrt(np.square(y_height / 2) + np.square(x_width))

    delta_x0 = np.array([2 * x_width, 0])
    delta_x1 = np.array([0.5 * delta_x0[0], np.sqrt(3) * delta_x0[0] / 2])

    if not np.allclose(delta_x1[1], y_height / 2 + diagonal_length):
        raise AssertionError(  # noqa: TRY003
            f"{delta_x1[1]} not close to {y_height / 2 + diagonal_length}"  # noqa: EM102
        )
    return {
        "basis": StackedBasis(
            FundamentalPositionBasis(delta_x0, final_grid.shape[0]),
            FundamentalPositionBasis(delta_x1, final_grid.shape[1]),
            UnevenPotential3dZBasis(z_points - z_points[0]),
        ),
        "data": final_grid.ravel(),
    }


def get_reflected_potential() -> UnevenPotential3d[int, int, int]:
    irreducible_points = load_raw_data()
    return map_irreducible_points_into_unit_cell(irreducible_points)


_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)


def get_interpolated_potential(
    shape: tuple[_L0, _L1, _L2]
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[_L0],
        FundamentalPositionBasis3d[_L1],
        FundamentalPositionBasis3d[_L2],
    ]
]:
    data = get_reflected_potential()
    normalized = normalize_potential(data)

    # The Top site has such an insanely large energy
    # We must bring it down first
    # ! truncated = truncate_potential(normalized, cutoff=1e-17, n=5, offset=1e-20)
    truncated = truncate_potential(normalized, cutoff=0.5e-18, n=1, offset=0)
    interpolated = interpolate_uneven_potential(truncated, shape)
    return undo_truncate_potential(interpolated, cutoff=0.5e-18, n=1, offset=0)
