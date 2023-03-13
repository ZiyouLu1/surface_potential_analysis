from functools import cached_property
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from surface_potential_analysis.brillouin_zone import grid_space


class SurfaceConfig(TypedDict):
    delta_x0: tuple[float, float]
    """lattice vector in the x0 direction"""
    delta_x1: tuple[float, float]
    """lattice vector in the x1 direction"""


def get_surface_xy_points(
    surface: SurfaceConfig,
    shape: tuple[int, int],
    *,
    offset: tuple[float, float] = (0.0, 0.0)
) -> NDArray:
    xy_points = grid_space(
        surface["delta_x0"], surface["delta_x1"], shape=shape, endpoint=False
    )
    xy_points[:, 0] += offset[0]
    xy_points[:, 1] += offset[1]
    return xy_points.reshape(*shape, 2)


def get_reciprocal_surface(surface: SurfaceConfig) -> SurfaceConfig:
    util = SurfaceConfigUtil(surface)
    return {"delta_x0": util.dkx0, "delta_x1": util.dkx1}


def get_surface_coordinates(
    surface: SurfaceConfig,
    shape: tuple[int, int],
    z_points: list[float],
    *,
    offset: tuple[float, float] = (0.0, 0.0)
) -> NDArray:
    xy_points = get_surface_xy_points(surface, shape, offset=offset)
    nz = len(z_points)

    tiled_x = np.tile(xy_points[:, :, 0], (nz, 1, 1)).swapaxes(0, 1).swapaxes(1, 2)
    tiled_y = np.tile(xy_points[:, :, 1], (nz, 1, 1)).swapaxes(0, 1).swapaxes(1, 2)
    tiled_z = np.tile(z_points, (*shape, 1))

    return (
        np.array([tiled_x + offset[0], tiled_y + offset[1], tiled_z])
        .swapaxes(0, 1)
        .swapaxes(1, 2)
        .swapaxes(2, 3)
    )


class SurfaceConfigUtil:

    _config: SurfaceConfig

    def __init__(self, config: SurfaceConfig) -> None:
        self._config = config

    @cached_property
    def _dk_prefactor(self):
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        x1_part = self.delta_x0[0] * self.delta_x1[1]
        x2_part = self.delta_x0[1] * self.delta_x1[0]
        return (2 * np.pi) / (x1_part - x2_part)

    @property
    def delta_x0(self) -> tuple[float, float]:
        return self._config["delta_x0"]

    @cached_property
    def dkx0(self) -> tuple[float, float]:
        return (
            self._dk_prefactor * self.delta_x1[1],
            -self._dk_prefactor * self.delta_x1[0],
        )

    @property
    def delta_x1(self) -> tuple[float, float]:
        return self._config["delta_x1"]

    @cached_property
    def dkx1(self) -> tuple[float, float]:
        return (
            -self._dk_prefactor * self.delta_x0[1],
            self._dk_prefactor * self.delta_x0[0],
        )
