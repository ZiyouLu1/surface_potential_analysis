from __future__ import annotations

from functools import cached_property
from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis import (
    Basis,
    BasisUtil,
    BasisVector,
    FundamentalBasis,
    MomentumBasis,
    PositionBasis,
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


# ruff: noqa: D102
class BasisConfigUtil(Generic[_BX0Cov, _BX1Cov, _BX2Cov]):
    """A class to help with the manipulation of basis configs."""

    _config: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]

    def __init__(self, config: BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]) -> None:
        self._config = config

    @cached_property
    def utils(
        self,
    ) -> tuple[BasisUtil[_BX0Cov], BasisUtil[_BX1Cov], BasisUtil[_BX2Cov]]:
        return (self.x0_basis, self.x1_basis, self.x2_basis)

    @cached_property
    def volume(self) -> float:
        out = np.dot(self.delta_x0, np.cross(self.delta_x1, self.delta_x2))
        assert out != 0
        return out  # type:ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> float:
        out = np.dot(self.dk0, np.cross(self.dk1, self.dk2))
        assert out != 0
        return out  # type:ignore[no-any-return]

    @property
    def fundamental_nk_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.fundamental_nk_points,  # type: ignore[misc]
            self.x1_basis.fundamental_nk_points,  # type: ignore[misc]
            self.x2_basis.fundamental_nk_points,  # type: ignore[misc]
            indexing="ij",
        )
        return np.array(  # type:ignore[no-any-return]
            [x0t.ravel(), x1t.ravel(), x2t.ravel()]
        )

    @property
    def fundamental_k_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        nk_points = self.fundamental_nk_points[:, np.newaxis, :]
        basis_vectors = np.array([self.dk0, self.dk1, self.dk2])[:, :, np.newaxis]
        return np.sum(basis_vectors * nk_points, axis=0)  # type: ignore[no-any-return]

    @property
    def fundamental_nx_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        x0t, x1t, x2t = np.meshgrid(
            self.x0_basis.fundamental_nx_points,  # type: ignore[misc]
            self.x1_basis.fundamental_nx_points,  # type: ignore[misc]
            self.x2_basis.fundamental_nx_points,  # type: ignore[misc]
            indexing="ij",
        )
        return np.array(  # type:ignore[no-any-return]
            [x0t.ravel(), x1t.ravel(), x2t.ravel()]
        )

    @property
    def fundamental_x_points(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        nx_points = self.fundamental_nx_points[:, np.newaxis, :]
        basis_vectors = np.array(
            [self.fundamental_dx0, self.fundamental_dx1, self.fundamental_dx2]
        )[:, :, np.newaxis]
        return np.sum(basis_vectors * nx_points, axis=0)  # type: ignore[no-any-return]

    @cached_property
    def x0_basis(self) -> BasisUtil[_BX0Cov]:
        return BasisUtil(self._config[0])

    @property
    def fundamental_n0(self) -> int:
        return self.x0_basis.fundamental_n  # type: ignore[misc]

    @property
    def n0(self) -> int:
        return self.x0_basis.n  # type: ignore[misc]

    @property
    def delta_x0(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self.x0_basis.delta_x

    @cached_property
    def fundamental_dx0(self) -> BasisVector:
        return self.delta_x0 / self.fundamental_n0  # type: ignore[no-any-return]

    @cached_property
    def fundamental_delta_k0(self) -> BasisVector:
        return self.fundamental_n0 * self.dk0  # type: ignore[no-any-return]

    @cached_property
    def dk0(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        return (  # type:ignore[no-any-return]
            2 * np.pi * np.cross(self.delta_x1, self.delta_x2) / self.volume
        )

    @cached_property
    def x1_basis(self) -> BasisUtil[_BX1Cov]:
        return BasisUtil(self._config[1])

    @property
    def fundamental_n1(self) -> int:
        return self.x1_basis.fundamental_n  # type: ignore[misc]

    @property
    def n1(self) -> int:
        return self.x1_basis.n  # type: ignore[misc]

    @property
    def delta_x1(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self.x1_basis.delta_x

    @cached_property
    def fundamental_dx1(self) -> BasisVector:
        return self.delta_x1 / self.fundamental_n1  # type: ignore[no-any-return]

    @cached_property
    def fundamental_delta_k1(self) -> BasisVector:
        return self.fundamental_n1 * self.dk1  # type: ignore[no-any-return]

    @cached_property
    def dk1(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        out = 2 * np.pi * np.cross(self.delta_x2, self.delta_x0) / self.volume
        return out  # type:ignore[no-any-return]  # noqa: RET504

    @cached_property
    def x2_basis(self) -> BasisUtil[_BX2Cov]:
        return BasisUtil(self._config[2])

    @property
    def fundamental_n2(self) -> int:
        return self.x2_basis.fundamental_n  # type: ignore[misc]

    @property
    def n2(self) -> int:
        return self.x2_basis.n  # type: ignore[misc]

    @property
    def delta_x2(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        return self.x2_basis.delta_x

    @cached_property
    def fundamental_dx2(self) -> BasisVector:
        return self.delta_x2 / self.fundamental_n2  # type: ignore[no-any-return]

    @cached_property
    def fundamental_delta_k2(self) -> BasisVector:
        return self.fundamental_n2 * self.dk2  # type: ignore[no-any-return]

    @cached_property
    def dk2(self) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        return (  # type:ignore[no-any-return]
            2 * np.pi * np.cross(self.delta_x0, self.delta_x1) / self.volume
        )

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        return (self.x0_basis.n, self.x1_basis.n, self.x2_basis.n)  # type: ignore[misc]

    @property
    def size(self) -> int:
        return np.prod(self.shape)  # type: ignore[return-value]

    @cached_property
    def fundamental_shape(self) -> tuple[int, int, int]:
        return (self.x0_basis.fundamental_n, self.x1_basis.fundamental_n, self.x2_basis.fundamental_n)  # type: ignore[misc]

    def __len__(self) -> int:
        return int(np.prod(self.shape))

    def get_flat_index(
        self,
        idx: tuple[int, int, int],
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> int:
        """
        Given a stacked index, get the flat index into the Wigner-Seitz cell.

        Parameters
        ----------
        idx : tuple[int, int, int]
            The stacked index
        mode : Literal[&quot;raise&quot;, &quot;wrap&quot;, &quot;clip&quot;], optional
            Specifies how out-of-bounds indices are handled, by default "raise"

        Returns
        -------
        int
            the flattened index into the Wigner-Seitz cell
        """
        return np.ravel_multi_index(idx, self.shape, mode=mode).item()

    def get_stacked_index(self, idx: int) -> tuple[int, int, int]:
        """
        Given a flat index, produce a stacked index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple[int, int, int]
        """
        stacked = np.unravel_index(idx, self.shape)
        return tuple([s.item() for s in stacked])  # type: ignore[return-value]

    def get_fundamental_basis(self) -> FundamentalBasisConfig[Any, Any, Any]:
        """
        Get the fundamental basis of the basis config.

        Returns
        -------
        FundamentalBasisConfig[Any, Any, Any]
        """
        return (
            self.x0_basis.fundamental_basis,
            self.x1_basis.fundamental_basis,
            self.x2_basis.fundamental_basis,
        )

    @overload
    def get_fundamental_basis_in(
        self, _type: Literal["position"]
    ) -> PositionBasisConfig[Any, Any, Any]:
        ...

    @overload
    def get_fundamental_basis_in(
        self, _type: Literal["momentum"]
    ) -> MomentumBasisConfig[Any, Any, Any]:
        ...

    def get_fundamental_basis_in(
        self, _type: Literal["position", "momentum"]
    ) -> FundamentalBasisConfig[Any, Any, Any]:
        """
        Get the fundamental basis of the given type.

        Parameters
        ----------
        _type : Literal[&quot;position&quot;, &quot;momentum&quot;]

        Returns
        -------
        FundamentalBasisConfig[Any, Any, Any]
        """
        return (  # type: ignore[return-value]
            {"_type": _type, "n": self.fundamental_n0, "delta_x": self.delta_x0},  # type: ignore[misc]
            {"_type": _type, "n": self.fundamental_n1, "delta_x": self.delta_x1},  # type: ignore[misc]
            {"_type": _type, "n": self.fundamental_n2, "delta_x": self.delta_x2},  # type: ignore[misc]
        )

    def get_rotated_basis(
        self,
        axis: Literal[0, 1, 2, -1, -2, -3] = 0,
        direction: BasisVector | None = None,
    ) -> BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]:
        """
        Get the basis, rotated such that axis is along the basis vector direction.

        Parameters
        ----------
        axis : Literal[0, 1, 2, -1, -2, -3], optional
            axis to point along the basis vector direction, by default 0
        direction : BasisVector | None, optional
            basis vector to point along, by default [0,0,1]

        Returns
        -------
        BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]
            _description_
        """
        matrix = _get_rotation_matrix(self.utils[axis].delta_x, direction)
        return (  # type: ignore[return-value]
            {
                **self._config[0],
                **(
                    {"delta_x": np.dot(matrix, self._config[0]["delta_x"])}  # type: ignore[typeddict-item]
                    if self._config[0].get("parent", None) is None
                    else {
                        "parent": {
                            **self._config[0]["parent"],  # type: ignore[typeddict-item]
                            "delta_x": np.dot(
                                matrix, self._config[0]["parent"]["delta_x"]  # type: ignore[typeddict-item]
                            ),
                        }
                    }
                ),
            },
            {
                **self._config[1],
                **(
                    {"delta_x": np.dot(matrix, self._config[1]["delta_x"])}  # type: ignore[typeddict-item]
                    if self._config[1].get("parent", None) is None
                    else {
                        "parent": {
                            **self._config[1]["parent"],  # type: ignore[typeddict-item]
                            "delta_x": np.dot(
                                matrix, self._config[1]["parent"]["delta_x"]  # type: ignore[typeddict-item]
                            ),
                        }
                    }
                ),
            },
            {
                **self._config[2],
                **(
                    {"delta_x": np.dot(matrix, self._config[2]["delta_x"])}  # type: ignore[typeddict-item]
                    if self._config[2].get("parent", None) is None
                    else {
                        "parent": {
                            **self._config[2]["parent"],  # type: ignore[typeddict-item]
                            "delta_x": np.dot(
                                matrix, self._config[2]["parent"]["delta_x"]  # type: ignore[typeddict-item]
                            ),
                        }
                    }
                ),
            },
        )


def _get_rotation_matrix(
    vector: BasisVector, direction: BasisVector | None = None
) -> np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float_]]:
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    unit = (
        np.array([0.0, 0, 1])
        if direction is None
        else direction.copy() / np.linalg.norm(direction)
    )
    # Normalize vector length
    vector = vector.copy() / np.linalg.norm(vector)

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
    return (  # type: ignore[no-any-return]
        rcos * np.eye(3)
        + rsin * np.array([[0, -w, v], [w, 0, -u], [-v, u, 0]])
        + (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )


class MomentumBasisConfigUtil(
    BasisConfigUtil[MomentumBasis[_L0Cov], MomentumBasis[_L1Cov], MomentumBasis[_L2Cov]]
):
    """A class to help with the manipulation of Momentum basis configs."""

    @staticmethod
    def from_resolution(
        resolution: tuple[_L0Cov, _L1Cov, _L2Cov],
        delta_x: tuple[BasisVector, BasisVector, BasisVector] | None = None,
    ) -> MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        """
        Get a momentum basis from a given delta_x and resolution.

        Parameters
        ----------
        resolution : tuple[_L0Cov, _L1Cov, _L2Cov]
        delta_x : tuple[BasisVector, BasisVector, BasisVector] | None, optional

        Returns
        -------
        MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]
        """
        delta_x = (
            (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
            if delta_x is None
            else delta_x
        )
        return (
            {"_type": "momentum", "n": resolution[0], "delta_x": delta_x[0]},
            {"_type": "momentum", "n": resolution[1], "delta_x": delta_x[1]},
            {"_type": "momentum", "n": resolution[2], "delta_x": delta_x[2]},
        )

    def get_reciprocal_basis(self) -> PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        """
        Get the reciprocal basis.

        Returns
        -------
        PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
        """
        return (
            {
                "_type": "position",
                "n": len(self.x0_basis),  # type:ignore[typeddict-item]
                "delta_x": self.delta_x0,
            },
            {
                "_type": "position",
                "n": len(self.x1_basis),  # type:ignore[typeddict-item]
                "delta_x": self.delta_x1,
            },
            {
                "_type": "position",
                "n": len(self.delta_x2),  # type:ignore[typeddict-item]
                "delta_x": self.delta_x2,
            },
        )


class PositionBasisConfigUtil(
    BasisConfigUtil[PositionBasis[_L0Cov], PositionBasis[_L1Cov], PositionBasis[_L2Cov]]
):
    """A class to help with the manipulation of position basis configs."""

    @staticmethod
    def from_resolution(
        resolution: tuple[_L0Cov, _L1Cov, _L2Cov],
        delta_x: tuple[BasisVector, BasisVector, BasisVector] | None = None,
    ) -> PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]:
        """
        Get a position basis from a given delta_x and resolution.

        Parameters
        ----------
        resolution : tuple[_L0Cov, _L1Cov, _L2Cov]
        delta_x : tuple[BasisVector, BasisVector, BasisVector] | None, optional

        Returns
        -------
        PositionBasisConfig[_L0Cov, _L1Cov, _L2Cov]
        """
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
        """
        Get the reciprocal basis.

        Returns
        -------
        MomentumBasisConfig[_L0Cov, _L1Cov, _L2Cov]
        """
        return (
            {
                "_type": "momentum",
                "n": len(self.x0_basis),  # type:ignore[typeddict-item]
                "delta_x": self.delta_x0,
            },
            {
                "_type": "momentum",
                "n": len(self.x1_basis),  # type:ignore[typeddict-item]
                "delta_x": self.delta_x1,
            },
            {
                "_type": "momentum",
                "n": len(self.x2_basis),  # type:ignore[typeddict-item]
                "delta_x": self.delta_x2,
            },
        )


def get_fundamental_projected_k_points(
    basis: BasisConfig[Any, Any, Any],
    axis: Literal[0, 1, 2, -1, -2, -3],
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    """
    Get a grid of points projected perpendicular to the given basis axis.

    This throws away the componet of the cooridnate grid in the direction
    parallel to axis.

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
    rotated = BasisConfigUtil(basis).get_rotated_basis(axis)
    util = BasisConfigUtil(rotated)
    return util.fundamental_k_points.reshape(3, *util.fundamental_shape)[0:2,]


def get_fundamental_projected_x_points(
    basis: BasisConfig[Any, Any, Any],
    axis: Literal[0, 1, 2, -1, -2, -3],
) -> np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]:
    """
    Get a grid of points projected perpendicular to the given basis axis, at the given index along this axis.

    This throws away the componet of the cooridnate grid in the direction
    parallel to axis.

    Parameters
    ----------
    basis : BasisConfig[Any, Any, Any]
    axis : Literal[0, 1, 2,-1, -2, -3]
        The index along the axis to take the coordinates from

    Returns
    -------
    np.ndarray[tuple[Literal[2], int, int], np.dtype[np.float_]]
        The coordinates in the plane perpendicular to axis
    """
    rotated = BasisConfigUtil(basis).get_rotated_basis(axis, np.array([0, 0, 1]))
    util = BasisConfigUtil(rotated)
    return util.fundamental_x_points.reshape(3, *util.fundamental_shape)[0:2]
