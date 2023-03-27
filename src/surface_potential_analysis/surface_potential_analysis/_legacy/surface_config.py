from functools import cached_property
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from surface_potential_analysis._legacy.brillouin_zone import grid_space


class SurfaceConfig(TypedDict):
    delta_x0: tuple[float, float]
    """lattice vector in the x0 direction"""
    delta_x1: tuple[float, float]
    """lattice vector in the x1 direction"""


class SurfaceConfigNew(TypedDict):
    delta_x0: tuple[float, float, float]
    """lattice vector in the x0 direction"""
    delta_x1: tuple[float, float, float]
    """lattice vector in the x1 direction"""
    delta_x2: tuple[float, float, float]
    """lattice vector in the x2 direction"""
    resolution: tuple[int, int, int]
    """Resolution in x0,x1,x2"""


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


class SurfaceConfigUtilNew:
    _config: SurfaceConfigNew

    def __init__(self, config: SurfaceConfigNew) -> None:
        self._config = config

    @property
    def resolution(self):
        return self._config["resolution"]

    @cached_property
    def surface_volume(self) -> float:
        return np.dot(self.delta_x0, np.cross(self.delta_x1, self.delta_x2))

    @property
    def delta_x0(self) -> tuple[float, float, float]:
        return self._config["delta_x0"]

    @property
    def Nx0(self) -> int:
        return self.resolution[0]

    @cached_property
    def dkx0(self) -> tuple[float, float, float]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        dkx0 = 2 * np.pi * np.cross(self.delta_x1, self.delta_x2) / self.surface_volume
        return (dkx0[0], dkx0[1], dkx0[2])

    @property
    def Nkx0(self) -> int:
        return self.resolution[0]

    @property
    def delta_x1(self) -> tuple[float, float, float]:
        return self._config["delta_x1"]

    @property
    def Nx1(self) -> int:
        return self.resolution[1]

    @cached_property
    def dkx1(self) -> tuple[float, float, float]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        dkx1 = 2 * np.pi * np.cross(self.delta_x2, self.delta_x1) / self.surface_volume
        return (dkx1[0], dkx1[1], dkx1[2])

    @property
    def Nkx1(self) -> int:
        return self.resolution[1]

    @property
    def delta_x2(self) -> tuple[float, float, float]:
        return self._config["delta_x2"]

    @property
    def Nx2(self) -> int:
        return self.resolution[1]

    @cached_property
    def dkx2(self) -> tuple[float, float, float]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        dkx2 = 2 * np.pi * np.cross(self.delta_x0, self.delta_x1) / self.surface_volume
        return (dkx2[0], dkx2[1], dkx2[2])

    @property
    def Nkx2(self) -> int:
        return self.resolution[1]

    def get_reciprocal(self) -> SurfaceConfigNew:
        return {
            "delta_x0": self.dkx0,
            "delta_x1": self.dkx1,
            "delta_x2": self.dkx2,
            "resolution": self.resolution,
        }


class SurfaceConfigUtil:
    util: SurfaceConfigUtilNew
    _config: SurfaceConfig

    def __init__(self, config: SurfaceConfig) -> None:
        self._config = config
        self.util = SurfaceConfigUtilNew(
            {
                "delta_x0": (config["delta_x0"][0], config["delta_x0"][1], 0),
                "delta_x1": (config["delta_x1"][0], config["delta_x1"][1], 0),
                "delta_x2": (0, 0, 1),
                "resolution": (0, 0, 0),
            }
        )

    @property
    def delta_x0(self) -> tuple[float, float]:
        delta_x0 = self.util.delta_x0
        return (delta_x0[0], delta_x0[1])

    @cached_property
    def dkx0(self) -> tuple[float, float]:
        dkx0 = self.util.dkx0
        return (dkx0[0], dkx0[1])

    @property
    def delta_x1(self) -> tuple[float, float]:
        delta_x1 = self.util.delta_x1
        return (delta_x1[0], delta_x1[1])

    @cached_property
    def dkx1(self) -> tuple[float, float]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        dkx1 = self.util.dkx1
        return (dkx1[0], dkx1[1])
