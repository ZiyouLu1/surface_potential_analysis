from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from surface_potential_analysis.basis import (
    Basis,
    BasisUtil,
    BasisVector,
    FundamentalBasis,
    MomentumBasis,
    MomentumBasisUtil,
    PositionBasis,
    get_fundamental_basis,
)

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)

_L0Cov = TypeVar("_L0Cov", bound=int, covariant=True)
_L1Cov = TypeVar("_L1Cov", bound=int, covariant=True)
_L2Cov = TypeVar("_L2Cov", bound=int, covariant=True)


BasisConfig = tuple[_BX0Cov, _BX1Cov, _BX2Cov]


MomentumBasisConfig = BasisConfig[
    MomentumBasis[_L0Cov], MomentumBasis[_L1Cov], MomentumBasis[_L2Cov]
]

PositionBasisConfig = BasisConfig[
    PositionBasis[_L0Cov], PositionBasis[_L1Cov], PositionBasis[_L2Cov]
]

FundamentalBasisConfig = BasisConfig[
    FundamentalBasis[_L0Cov], FundamentalBasis[_L1Cov], FundamentalBasis[_L2Cov]
]


class BasisConfigUtil(Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    _config: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]

    def __init__(self, config: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]) -> None:
        self._config = config

    @cached_property
    def x0_basis(self) -> BasisUtil[_BX0Cov]:
        return BasisUtil(self._config[0])

    @cached_property
    def x1_basis(self) -> BasisUtil[_BX1Cov]:
        return BasisUtil(self._config[1])

    @cached_property
    def x2_basis(self) -> BasisUtil[_BX2Cov]:
        return BasisUtil(self._config[2])

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.x0_basis), len(self.x1_basis), len(self.x2_basis))

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def get_flat_index(self, idx: tuple[int, int, int]) -> int:
        return np.ravel_multi_index(idx, self.shape).item()

    def get_stacked_index(self, idx: int) -> tuple[int, int, int]:
        stacked = np.unravel_index(idx, self.shape)
        return tuple([s.item() for s in stacked])  # type: ignore

    def get_fundamental_basis(self) -> FundamentalBasisConfig[Any, Any, Any]:
        return tuple([get_fundamental_basis(b) for b in self._config])  # type: ignore


_FBX0 = TypeVar("_FBX0", bound=FundamentalBasis[Any])
_FBX1 = TypeVar("_FBX1", bound=FundamentalBasis[Any])
_FBX2 = TypeVar("_FBX2", bound=FundamentalBasis[Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)


class FundamentalBasisConfigUtil(ABC, BasisConfigUtil[_FBX0, _FBX1, _FBX2]):
    @cached_property
    def volume(self) -> float:
        out = np.dot(self.delta_x0, np.cross(self.delta_x1, self.delta_x2))
        return out  # type:ignore

    @cached_property
    def reciprocal_volume(self) -> float:
        out = np.dot(self.dk0, np.cross(self.dk1, self.dk2))
        return out  # type:ignore

    @property
    def nk_points(self) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.nk_points,  # type: ignore
            self.x1_basis.nk_points,  # type: ignore
            self.x2_basis.nk_points,  # type: ignore
            indexing="ij",
        )
        return np.array([x0t.ravel(), x1t.ravel(), x2t.ravel()])  # type:ignore

    @property
    def k_points(self) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        nk_points = self.nk_points[:, np.newaxis, :]
        basis_vectors = np.array([self.dk0, self.dk1, self.dk2])[:, :, np.newaxis]
        return np.sum(basis_vectors * nk_points, axis=0)  # type: ignore

    @property
    def nx_points(self) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.nx_points,  # type: ignore
            self.x1_basis.nx_points,  # type: ignore
            self.x2_basis.nx_points,  # type: ignore
            indexing="ij",
        )
        return np.array([x0t.ravel(), x1t.ravel(), x2t.ravel()])  # type:ignore

    @property
    def x_points(self) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        nx_points = self.nx_points[:, np.newaxis, :]
        basis_vectors = np.array([self.dx0, self.dx1, self.dx2])[:, :, np.newaxis]
        return np.sum(basis_vectors * nx_points, axis=0)  # type: ignore

    @property
    def Nx0(self) -> int:
        return self.x0_basis.Nx  # type: ignore

    @property
    def Nk0(self) -> int:
        return self.x0_basis.Nk  # type: ignore

    @property
    @abstractmethod
    def delta_x0(self) -> BasisVector:
        ...

    @cached_property
    def dx0(self) -> BasisVector:
        return self.delta_x0 / self.Nx0  # type: ignore

    @cached_property
    def delta_k0(self) -> BasisVector:
        return self.Nk0 * self.dk0  # type: ignore

    @property
    @abstractmethod
    def dk0(self) -> BasisVector:
        ...

    @property
    def Nx1(self) -> int:
        return self.x1_basis.Nx  # type: ignore

    @property
    def Nk1(self) -> int:
        return self.x1_basis.Nk  # type: ignore

    @property
    @abstractmethod
    def delta_x1(self) -> BasisVector:
        ...

    @cached_property
    def dx1(self) -> BasisVector:
        return self.delta_x1 / self.Nx1  # type: ignore

    @cached_property
    def delta_k1(self) -> BasisVector:
        return self.Nk1 * self.dk1  # type: ignore

    @property
    @abstractmethod
    def dk1(self) -> BasisVector:
        ...

    @property
    def Nx2(self) -> int:
        return self.x2_basis.Nx  # type: ignore

    @property
    def Nk2(self) -> int:
        return self.x2_basis.Nk  # type: ignore

    @property
    @abstractmethod
    def delta_x2(self) -> BasisVector:
        ...

    @cached_property
    def dx2(self) -> BasisVector:
        return self.delta_x2 / self.Nx2  # type: ignore

    @cached_property
    def delta_k2(self) -> BasisVector:
        return self.Nk2 * self.dk2  # type: ignore

    @property
    @abstractmethod
    def dk2(self) -> BasisVector:
        ...

    def get_fundamental_basis(self) -> BasisConfig[_FBX0, _FBX1, _FBX2]:
        return self._config


def _get_rotation_matrix(
    vector: BasisVector, direction: BasisVector | None = None
) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]:
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    unit = (
        np.array([0, 0, 1])
        if direction is None
        else direction / np.linalg.norm(direction)
    )
    # Normalize vector length
    vector /= np.linalg.norm(vector)

    # Get axis
    uvw = np.cross(vector, unit)

    # compute trig values - no need to go through arccos and back
    rcos: np.float_ = np.dot(vector, unit)
    rsin: np.float_ = np.linalg.norm(uvw)

    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (  # type: ignore
        rcos * np.eye(3)
        + rsin * np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
        + (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )


class MomentumBasisConfigUtil(
    FundamentalBasisConfigUtil[
        MomentumBasis[_L0Cov], MomentumBasis[_L1Cov], MomentumBasis[_L2Cov]
    ]
):
    @cached_property
    def x0_basis(self) -> MomentumBasisUtil[_L0Cov]:
        return MomentumBasisUtil(self._config[0])

    @cached_property
    def x1_basis(self) -> MomentumBasisUtil[_L1Cov]:
        return MomentumBasisUtil(self._config[1])

    @cached_property
    def x2_basis(self) -> MomentumBasisUtil[_L2Cov]:
        return MomentumBasisUtil(self._config[2])

    @property
    def delta_x0(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        out = 2 * np.pi * np.cross(self.dk1, self.dk2) / self.reciprocal_volume
        return out  # type:ignore

    @cached_property
    def dk0(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self._config[0]["dk"]

    @property
    def delta_x1(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        out = 2 * np.pi * np.cross(self.dk2, self.dk0) / self.reciprocal_volume
        return out  # type:ignore

    @cached_property
    def dk1(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self._config[1]["dk"]

    @property
    def delta_x2(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        out = 2 * np.pi * np.cross(self.dk0, self.dk1) / self.reciprocal_volume
        return out  # type:ignore

    @cached_property
    def dk2(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self._config[2]["dk"]

    @staticmethod
    def from_resolution(
        resolution: tuple[_L0Cov, _L1Cov, _L2Cov],
        dk: tuple[BasisVector, BasisVector, BasisVector] | None = None,
    ) -> MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        dk = (
            (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
            if dk is None
            else dk
        )
        return (
            {"_type": "momentum", "n": resolution[0], "dk": dk[0]},
            {"_type": "momentum", "n": resolution[1], "dk": dk[1]},
            {"_type": "momentum", "n": resolution[2], "dk": dk[2]},
        )

    def get_reciprocal_basis(self) -> PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        return (
            {
                "_type": "position",
                "n": len(self.x0_basis),  # type:ignore
                "delta_x": self.delta_x0,
            },
            {
                "_type": "position",
                "n": len(self.x1_basis),  # type:ignore
                "delta_x": self.delta_x1,
            },
            {
                "_type": "position",
                "n": len(self.delta_x2),  # type:ignore
                "delta_x": self.delta_x2,
            },
        )

    def get_rotated_basis(
        self,
        basis: Literal[0, 1, 2, -1, -2, -3] = 0,
        direction: BasisVector | None = None,
    ) -> PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        matrix = _get_rotation_matrix(self._config[basis]["dk"], direction)
        return (
            {
                "_type": "position",
                "n": self._config[0]["n"],
                "dk": np.dot(matrix, self._config[0]["dk"]),
            },
            {
                "_type": "position",
                "n": self._config[1]["n"],
                "dk": np.dot(matrix, self._config[1]["dk"]),
            },
            {
                "_type": "position",
                "n": self._config[2]["n"],
                "dk": np.dot(matrix, self._config[2]["dk"]),
            },
        )


class PositionBasisConfigUtil(
    FundamentalBasisConfigUtil[
        PositionBasis[_L0Cov], PositionBasis[_L1Cov], PositionBasis[_L2Cov]
    ]
):
    @property
    def delta_x0(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self._config[0]["delta_x"]

    @cached_property
    def dk0(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        out = 2 * np.pi * np.cross(self.delta_x1, self.delta_x2) / self.volume
        return out  # type:ignore

    @property
    def delta_x1(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self._config[1]["delta_x"]

    @cached_property
    def dk1(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        out = 2 * np.pi * np.cross(self.delta_x2, self.delta_x0) / self.volume
        return out  # type:ignore

    @property
    def delta_x2(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self._config[2]["delta_x"]

    @cached_property
    def dk2(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        out = 2 * np.pi * np.cross(self.delta_x0, self.delta_x1) / self.volume
        return out  # type:ignore

    @staticmethod
    def from_resolution(
        resolution: tuple[_L0Cov, _L1Cov, _L2Cov],
        delta_x: tuple[BasisVector, BasisVector, BasisVector] | None = None,
    ) -> PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        delta_x = (
            (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
            if delta_x is None
            else delta_x
        )
        return (
            {"_type": "position", "n": resolution[0], "delta_x": delta_x[0]},
            {"_type": "position", "n": resolution[1], "delta_x": delta_x[1]},
            {"_type": "position", "n": resolution[2], "delta_x": delta_x[2]},
        )

    def get_reciprocal_basis(self) -> MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        return (
            {
                "_type": "momentum",
                "n": len(self.x0_basis),  # type:ignore
                "dk": self.dk0,
            },
            {
                "_type": "momentum",
                "n": len(self.x1_basis),  # type:ignore
                "dk": self.dk1,
            },
            {
                "_type": "momentum",
                "n": len(self.delta_x2),  # type:ignore
                "dk": self.dk2,
            },
        )

    def get_rotated_basis(
        self,
        basis: Literal[0, 1, 2, -1, -2, -3] = 0,
        direction: BasisVector | None = None,
    ) -> PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        matrix = _get_rotation_matrix(self._config[basis]["delta_x"], direction)
        return (
            {
                "_type": "position",
                "n": self._config[0]["n"],
                "delta_x": np.dot(matrix, self._config[0]["delta_x"]),
            },
            {
                "_type": "position",
                "n": self._config[1]["n"],
                "delta_x": np.dot(matrix, self._config[1]["delta_x"]),
            },
            {
                "_type": "position",
                "n": self._config[2]["n"],
                "delta_x": np.dot(matrix, self._config[2]["delta_x"]),
            },
        )


def get_projected_k_points(
    basis: MomentumBasisConfig[_L0Inv, _L1Inv, _L2Inv],
    axis: Literal[0, 1, 2, -1, -2, -3],
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    """
    Get a grid of points projected perpendicular to the given basis axis
    at the given index along this axis
    This throws away the componet of the cooridnate grid in the direction
    parallel to axis

    Parameters
    ----------
    basis : MomentumBasisConfig[_L0, _L1, _L2]
    axis : Literal[0, 1, 2,-1, -2, -3]
        The index along the axis to take the coordinates from

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
        The coordinates in the plane perpendicular to axis
    """
    rotated = MomentumBasisConfigUtil(basis).get_rotated_basis(axis)
    util = PositionBasisConfigUtil(rotated)
    return util.k_points.reshape(3, *util.shape)[0:2,]


def get_projected_x_points(
    basis: PositionBasisConfig[_L0Inv, _L1Inv, _L2Inv],
    axis: Literal[0, 1, 2, -1, -2, -3],
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    """
    Get a grid of points projected perpendicular to the given basis axis
    at the given index along this axis
    This throws away the componet of the cooridnate grid in the direction
    parallel to axis

    Parameters
    ----------
    basis : PositionBasisConfig[_L0, _L1, _L2]
    axis : Literal[0, 1, 2,-1, -2, -3]
        The index along the axis to take the coordinates from

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
        The coordinates in the plane perpendicular to axis
    """
    rotated = PositionBasisConfigUtil(basis).get_rotated_basis(axis)
    util = PositionBasisConfigUtil(rotated)
    return util.x_points.reshape(3, *util.shape)[0:2,]
