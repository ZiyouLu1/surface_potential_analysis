from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Unpack, overload

import numpy as np

from surface_potential_analysis.axis.axis import (
    FundamentalPositionAxis3d,
)
from surface_potential_analysis.axis.conversion import get_rotated_axis
from surface_potential_analysis.axis.util import Axis3dUtil, AxisUtil
from surface_potential_analysis.util.util import (
    slice_ignoring_axes,
)

from .basis import Basis, Basis3d

if TYPE_CHECKING:
    from surface_potential_analysis._types import (
        ArrayFlatIndexLike,
        ArrayIndexLike,
        ArrayIndexLike3d,
        ArrayStackedIndexLike,
        ArrayStackedIndexLike3d,
        FlatIndexLike,
        IndexLike3d,
        SingleFlatIndexLike,
        SingleIndexLike,
        SingleIndexLike3d,
        SingleStackedIndexLike,
        SingleStackedIndexLike3d,
        StackedIndexLike,
        StackedIndexLike3d,
        _IntLike_co,
    )
    from surface_potential_analysis.axis.axis_like import (
        AxisLike3d,
        AxisVector3d,
    )

    _A3d0Inv = TypeVar("_A3d0Inv", bound=AxisLike3d[Any, Any])
    _A3d1Inv = TypeVar("_A3d1Inv", bound=AxisLike3d[Any, Any])
    _A3d2Inv = TypeVar("_A3d2Inv", bound=AxisLike3d[Any, Any])

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])
_B0Inv = TypeVar("_B0Inv", bound=Basis[Any])
_B3d0Inv = TypeVar("_B3d0Inv", bound=Basis3d[Any, Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)
_L2Inv = TypeVar("_L2Inv", bound=int)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)


def _get_rotation_matrix(
    vector: AxisVector3d, direction: AxisVector3d | None = None
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


def get_rotated_basis3d(
    basis: Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv],
    axis: Literal[0, 1, 2, -1, -2, -3] = 0,
    direction: AxisVector3d | None = None,
) -> Basis3d[_A3d0Inv, _A3d1Inv, _A3d2Inv]:
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
    Basis3d[_A3d0Cov, _A3d1Cov, _A3d2Cov]
        _description_
    """
    matrix = _get_rotation_matrix(basis[axis].delta_x, direction)
    return (
        get_rotated_axis(basis[0], matrix),
        get_rotated_axis(basis[1], matrix),
        get_rotated_axis(basis[2], matrix),
    )


class BasisUtil(Generic[_B0Inv]):
    """
    A class to help with the manipulation of basis states.

    Note: The dimension of the axes must match the number of axes
    """

    _basis: _B0Inv

    def __init__(self, basis: _B0Inv) -> None:
        if any(x.delta_x.size != len(basis) for x in basis):
            msg = "Basis has incorrect shape"
            raise AssertionError(msg)
        self._basis = basis

    @cached_property
    def _utils(self) -> tuple[AxisUtil[Any, Any, Any], ...]:
        return tuple(AxisUtil(b) for b in self._basis)

    @cached_property
    def volume(self) -> np.float_:
        return np.linalg.det(self.delta_x)  # type: ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> np.float_:
        return np.linalg.det(self.dk)  # type: ignore[no-any-return]

    @property
    def nk_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[xi_axis.nk_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @property
    def fundamental_nk_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nk_mesh = np.meshgrid(
            *[xi_axis.fundamental_nk_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nki.ravel() for nki in nk_mesh)

    @overload
    def get_k_points_at_index(
        self, idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        ...

    @overload
    def get_k_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv]
    ) -> np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]:
        ...

    def get_k_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv] | SingleIndexLike
    ) -> (
        np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
        | np.ndarray[tuple[int], np.dtype[np.float_]]
    ):
        nk_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dk, nk_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def k_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.get_k_points_at_index(self.nk_points)

    @property
    def fundamental_k_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.tensordot(self.fundamental_dk, self.fundamental_nk_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def nx_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[xi_axis.nx_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @property
    def fundamental_nx_points(self) -> ArrayStackedIndexLike[tuple[int]]:
        nx_mesh = np.meshgrid(
            *[xi_axis.fundamental_nx_points for xi_axis in self._utils],
            indexing="ij",
        )
        return tuple(nxi.ravel() for nxi in nx_mesh)

    @overload
    def get_x_points_at_index(
        self, idx: SingleIndexLike
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        ...

    @overload
    def get_x_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv]
    ) -> np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]:
        ...

    def get_x_points_at_index(
        self, idx: ArrayIndexLike[_S0Inv] | SingleIndexLike
    ) -> (
        np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
        | np.ndarray[tuple[int], np.dtype[np.float_]]
    ):
        nx_points = idx if isinstance(idx, tuple) else self.get_stacked_index(idx)
        return np.tensordot(self.dx, nx_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def x_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.get_x_points_at_index(self.nx_points)

    @property
    def fundamental_x_points(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.tensordot(self.fundamental_dx, self.fundamental_nx_points, axes=(0, 0))  # type: ignore[no-any-return]

    @property
    def delta_x(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.delta_x for axi in self._basis])  # type: ignore[no-any-return]

    @property
    def fundamental_delta_x(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.delta_x for axi in self._basis])  # type: ignore[no-any-return]

    @cached_property
    def dx(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.dx for axi in self._utils])  # type: ignore[no-any-return]

    @cached_property
    def fundamental_dx(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array([axi.fundamental_dx for axi in self._utils])  # type: ignore[no-any-return]

    @property
    def delta_k(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array(self.shape)[:, np.newaxis] * self.dk  # type: ignore[no-any-return]

    @cached_property
    def fundamental_delta_k(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return np.array(self.fundamental_shape)[:, np.newaxis] * self.dk  # type: ignore[no-any-return]

    @cached_property
    def dk(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        """Get dk as a list of dk for each axis."""
        return 2 * np.pi * np.linalg.inv(self.delta_x).T  # type: ignore[no-any-return]

    @property
    def fundamental_dk(self) -> np.ndarray[tuple[int, int], np.dtype[np.float_]]:
        return self.dk

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(axi.n for axi in self._basis)

    @property
    def size(self) -> int:
        return np.prod(self.shape)  # type: ignore[return-value]

    @cached_property
    def fundamental_shape(self) -> tuple[int, ...]:
        return tuple(axi.fundamental_n for axi in self._basis)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @overload
    def get_flat_index(
        self,
        idx: SingleStackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_:
        ...

    @overload
    def get_flat_index(
        self,
        idx: ArrayStackedIndexLike[_S0Inv],
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> ArrayFlatIndexLike[_S0Inv]:
        ...

    def get_flat_index(
        self,
        idx: StackedIndexLike,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_ | ArrayFlatIndexLike[_S0Inv]:
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
        return np.ravel_multi_index(idx, self.shape, mode=mode)

    @overload
    def get_stacked_index(self, idx: SingleFlatIndexLike) -> SingleStackedIndexLike:
        ...

    @overload
    def get_stacked_index(
        self, idx: ArrayFlatIndexLike[_S0Inv]
    ) -> ArrayStackedIndexLike[_S0Inv]:
        ...

    def get_stacked_index(self, idx: FlatIndexLike) -> StackedIndexLike:
        """
        Given a flat index, produce a stacked index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple[int, int, int]
        """
        return np.unravel_index(idx, self.shape)  # type: ignore[return-value]


# ruff: noqa: D102
class Basis3dUtil(BasisUtil[_B3d0Inv]):
    """A class to help with the manipulation of basis states in 3d."""

    @property
    def _utils(
        self,
    ) -> tuple[Axis3dUtil[Any, Any], Axis3dUtil[Any, Any], Axis3dUtil[Any, Any]]:
        return super()._utils  # type: ignore[return-value]

    @cached_property
    def volume(self) -> np.float_:
        out = np.dot(self.delta_x0, np.cross(self.delta_x1, self.delta_x2))
        assert out != 0
        return out  # type: ignore[no-any-return]

    @cached_property
    def reciprocal_volume(self) -> np.float_:
        out = np.dot(self.dk0, np.cross(self.dk1, self.dk2))
        assert out != 0
        return out  # type: ignore[no-any-return]

    @property
    def nk_points(self) -> ArrayStackedIndexLike3d[tuple[int]]:
        return super().nk_points  # type: ignore[return-value]

    @property
    def fundamental_nk_points(self) -> ArrayStackedIndexLike3d[tuple[int]]:
        return super().fundamental_nk_points  # type: ignore[return-value]

    @overload  # type: ignore[override]
    def get_k_points_at_index(
        self, idx: SingleIndexLike3d
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        ...

    @overload
    def get_k_points_at_index(
        self, idx: ArrayIndexLike3d[_S0Inv]
    ) -> np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]:
        ...

    # Liskov substitution principle invalidated as we are not able to specify
    # that this is only a supertype of the BasisUtil in 3d
    def get_k_points_at_index(
        self, idx: SingleIndexLike3d | ArrayIndexLike3d[_S0Inv]
    ) -> (
        np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]
        | np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ):
        return super().get_k_points_at_index(idx)  # type: ignore[return-value]

    @property
    def k_points(  # type: ignore[override]
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        return super().k_points  # type: ignore[return-value]

    @property
    def fundamental_k_points(  # type: ignore[override]
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        return super().fundamental_k_points  # type: ignore[return-value]

    @property
    def nx_points(self) -> ArrayStackedIndexLike3d[tuple[int]]:
        return super().nx_points  # type: ignore[return-value]

    @property
    def fundamental_nx_points(self) -> ArrayStackedIndexLike3d[tuple[int]]:
        return super().fundamental_nx_points  # type: ignore[return-value]

    @overload  # type: ignore[override]
    def get_x_points_at_index(
        self, idx: SingleIndexLike3d
    ) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]:
        ...

    @overload
    def get_x_points_at_index(
        self, idx: ArrayIndexLike3d[_S0Inv]
    ) -> np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]:
        ...

    def get_x_points_at_index(
        self, idx: SingleIndexLike3d | ArrayIndexLike3d[_S0Inv]
    ) -> (
        np.ndarray[tuple[Literal[3], Unpack[_S0Inv]], np.dtype[np.float_]]
        | np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ):
        return super().get_x_points_at_index(idx)  # type: ignore[return-value]

    @property
    def x_points(  # type: ignore[override]
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        return super().x_points  # type: ignore[return-value]

    @property
    def fundamental_x_points(  # type: ignore[override]
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.float_]]:
        return super().fundamental_x_points  # type: ignore[return-value]

    @property
    def x0_basis(
        self: Basis3dUtil[tuple[AxisLike3d[_LF0Inv, _L0Inv], _A3d1Inv, _A3d2Inv]]
    ) -> Axis3dUtil[_LF0Inv, _L0Inv]:
        return self._utils[0]

    @property
    def fundamental_n0(
        self: Basis3dUtil[tuple[AxisLike3d[_LF0Inv, _L0Inv], _A3d1Inv, _A3d2Inv]]
    ) -> _LF0Inv:
        return self.fundamental_shape[0]  # type: ignore[return-value]

    @property
    def n0(
        self: Basis3dUtil[tuple[AxisLike3d[_LF0Inv, _L0Inv], _A3d1Inv, _A3d2Inv]]
    ) -> _L0Inv:
        return self.shape[0]  # type: ignore[return-value]

    @property
    def delta_x0(self) -> AxisVector3d:
        return self.delta_x[0]

    @property
    def fundamental_delta_x0(self) -> AxisVector3d:
        return self.fundamental_delta_x[0]

    @cached_property
    def dx0(self) -> AxisVector3d:
        return self.dx[0]

    @cached_property
    def fundamental_dx0(self) -> AxisVector3d:
        return self.fundamental_dx[0]

    @property
    def delta_k0(self) -> AxisVector3d:
        return self.delta_k[0]

    @cached_property
    def fundamental_delta_k0(self) -> AxisVector3d:
        return self.fundamental_delta_k[0]

    @cached_property
    def dk0(self) -> AxisVector3d:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        return (  # type: ignore[no-any-return]
            2 * np.pi * np.cross(self.delta_x1, self.delta_x2) / self.volume
        )

    @property
    def fundamental_dk0(self) -> AxisVector3d:
        return self.fundamental_dk[0]

    @property
    def x1_basis(
        self: Basis3dUtil[tuple[AxisLike3d[_LF1Inv, _L1Inv], _A3d1Inv, _A3d2Inv]]
    ) -> Axis3dUtil[_LF1Inv, _L1Inv]:
        return self._utils[1]

    @property
    def fundamental_n1(
        self: Basis3dUtil[tuple[AxisLike3d[_LF1Inv, _L1Inv], _A3d1Inv, _A3d2Inv]]
    ) -> _LF1Inv:
        return self.fundamental_shape[1]  # type: ignore[return-value]

    @property
    def n1(
        self: Basis3dUtil[tuple[AxisLike3d[_LF1Inv, _L1Inv], _A3d1Inv, _A3d2Inv]]
    ) -> _L1Inv:
        return self.shape[1]  # type: ignore[return-value]

    @property
    def delta_x1(self) -> AxisVector3d:
        return self.delta_x[1]

    @property
    def fundamental_delta_x1(self) -> AxisVector3d:
        return self.fundamental_delta_x[1]

    @cached_property
    def dx1(self) -> AxisVector3d:
        return self.dx[1]

    @cached_property
    def fundamental_dx1(self) -> AxisVector3d:
        return self.fundamental_dx[1]

    @property
    def delta_k1(self) -> AxisVector3d:
        return self.delta_k[1]

    @cached_property
    def fundamental_delta_k1(self) -> AxisVector3d:
        return self.fundamental_delta_k[1]

    @cached_property
    def dk1(self) -> AxisVector3d:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        out = 2 * np.pi * np.cross(self.delta_x2, self.delta_x0) / self.volume
        return out  # type: ignore[no-any-return]  # noqa: RET504

    @property
    def fundamental_dk1(self) -> AxisVector3d:
        return self.fundamental_dk[1]

    @property
    def x2_basis(
        self: Basis3dUtil[tuple[_A3d0Inv, _A3d1Inv, AxisLike3d[_LF2Inv, _L2Inv]]]
    ) -> Axis3dUtil[_LF2Inv, _L2Inv]:
        return self._utils[2]

    @property
    def fundamental_n2(
        self: Basis3dUtil[tuple[_A3d0Inv, _A3d1Inv, AxisLike3d[_LF2Inv, _L2Inv]]]
    ) -> int:
        return self.fundamental_shape[2]

    @property
    def n2(
        self: Basis3dUtil[tuple[_A3d0Inv, _A3d1Inv, AxisLike3d[_LF2Inv, _L2Inv]]]
    ) -> int:
        return self.shape[2]

    @property
    def delta_x2(self) -> AxisVector3d:
        return self.delta_x[2]

    @property
    def fundamental_delta_x2(self) -> AxisVector3d:
        return self.fundamental_delta_x[2]

    @cached_property
    def dx2(self) -> AxisVector3d:
        return self.dx[2]

    @cached_property
    def fundamental_dx2(self) -> AxisVector3d:
        return self.fundamental_dx[2]

    @property
    def delta_k2(self) -> AxisVector3d:
        return self.delta_k[2]

    @cached_property
    def fundamental_delta_k2(self) -> AxisVector3d:
        return self.fundamental_delta_k[2]

    @cached_property
    def dk2(self) -> AxisVector3d:
        # See https://physics.stackexchange.com/questions/340860/reciprocal-lattice-in-2d
        return (  # type: ignore[no-any-return]
            2 * np.pi * np.cross(self.delta_x0, self.delta_x1) / self.volume
        )

    @property
    def fundamental_dk2(self) -> AxisVector3d:
        return self.dk[2]

    @cached_property
    def shape(self) -> tuple[int, int, int]:
        return super().shape  # type: ignore[return-value]

    @property
    def size(self) -> int:
        return super().size

    @cached_property
    def fundamental_shape(self) -> tuple[int, int, int]:
        return super().fundamental_shape  # type: ignore[return-value]

    @property
    def ndim(self) -> Literal[3]:
        return 3

    @overload  # type: ignore[override]
    def get_flat_index(
        self,
        idx: SingleStackedIndexLike3d,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_:
        ...

    @overload
    def get_flat_index(
        self,
        idx: ArrayStackedIndexLike3d[_S0Inv],
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> ArrayFlatIndexLike[_S0Inv]:
        ...

    def get_flat_index(
        self,
        idx: StackedIndexLike3d,
        *,
        mode: Literal["raise", "wrap", "clip"] = "raise",
    ) -> np.int_ | ArrayFlatIndexLike[_S0Inv]:
        return super().get_flat_index(idx, mode=mode)

    @overload
    def get_stacked_index(self, idx: SingleFlatIndexLike) -> SingleStackedIndexLike3d:
        ...

    @overload
    def get_stacked_index(
        self, idx: ArrayFlatIndexLike[_S0Inv]
    ) -> ArrayStackedIndexLike3d[_S0Inv]:
        ...

    def get_stacked_index(self, idx: FlatIndexLike) -> StackedIndexLike3d:
        """
        Given a flat index, produce a stacked index.

        Parameters
        ----------
        idx : int

        Returns
        -------
        tuple[int, int, int]
        """
        return super().get_stacked_index(idx)  # type: ignore[return-value]


def project_k_points_along_axes(
    points: np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]],
    basis: _B0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]:
    """
    Get the list of k points projected onto the plane including both axes.

    Parameters
    ----------
    points : np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)

    ax_0 = util.delta_k[axes[0]] / np.linalg.norm(util.delta_k[axes[0]])
    # Subtract off parallel componet
    ax_1 = util.delta_k[axes[1]] - np.tensordot(ax_0, util.delta_k[axes[1]], 1)
    ax_1 /= np.linalg.norm(ax_1)

    projected_0 = np.tensordot(ax_0, points, axes=(0, 0))
    projected_1 = np.tensordot(ax_1, points, axes=(0, 0))

    return np.array([projected_0, projected_1])  # type: ignore[no-any-return]


def get_fundamental_k_points_projected_along_axes(
    basis: _B0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)
    points = util.fundamental_k_points
    return project_k_points_along_axes(points, basis, axes)


def get_k_coordinates_in_axes(
    basis: _B0Inv,
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None,
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)
    idx = tuple(0 for _ in range(util.ndim - len(axes))) if idx is None else idx
    points = get_fundamental_k_points_projected_along_axes(basis, axes)
    _slice = slice_ignoring_axes(idx, axes)
    return points.reshape(2, *util.shape)[:, *_slice]


def project_x_points_along_axes(
    points: np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]],
    basis: _B0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]:
    """
    Get the list of x points projected onto the plane including both axes.

    Parameters
    ----------
    points : np.ndarray[tuple[int, Unpack[_S0Inv]], np.dtype[np.float_]]
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], Unpack[_S0Inv]], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)

    ax_0 = util.delta_x[axes[0]] / np.linalg.norm(util.delta_x[axes[0]])
    # Subtract off parallel componet
    ax_1 = util.delta_x[axes[1]] - np.tensordot(ax_0, util.delta_x[axes[1]], 1)
    ax_1 /= np.linalg.norm(ax_1)

    projected_0 = np.tensordot(ax_0, points, axes=(0, 0))
    projected_1 = np.tensordot(ax_1, points, axes=(0, 0))

    return np.array([projected_0, projected_1])  # type: ignore[no-any-return]


def get_fundamental_x_points_projected_along_axes(
    basis: _B0Inv,
    axes: tuple[int, int],
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_x_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)
    points = util.fundamental_x_points
    return project_x_points_along_axes(points, basis, axes)


def get_x_coordinates_in_axes(
    basis: _B0Inv,
    axes: tuple[int, int],
    idx: SingleStackedIndexLike | None,
) -> np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]:
    """
    Get the fundamental_k_points projected onto the plane including both axes.

    Parameters
    ----------
    basis : _B0Inv
    axes : tuple[int, int]

    Returns
    -------
    np.ndarray[tuple[Literal[2], int], np.dtype[np.float_]]
    """
    util = BasisUtil(basis)
    idx = tuple(0 for _ in range(util.ndim - len(axes))) if idx is None else idx
    points = get_fundamental_x_points_projected_along_axes(basis, axes)
    _slice = slice_ignoring_axes(idx, axes)
    return points.reshape(2, *util.shape)[:, *_slice]


@overload
def _wrap_distance(distance: _IntLike_co, length: int) -> int:
    ...


@overload
def _wrap_distance(
    distance: np.ndarray[_S0Inv, np.dtype[np.int_]], length: int
) -> np.ndarray[_S0Inv, np.dtype[np.int_]]:
    ...


def _wrap_distance(distance: Any, length: int) -> Any:
    return np.subtract(np.mod(np.add(distance, length // 2), length), length // 2)


@overload
def wrap_index_around_origin_x01(
    basis: _B3d0Inv,
    idx: SingleStackedIndexLike3d,
    origin_idx: SingleIndexLike3d = (0, 0, 0),
) -> SingleStackedIndexLike3d:
    ...


@overload
def wrap_index_around_origin_x01(
    basis: _B3d0Inv, idx: SingleFlatIndexLike, origin_idx: SingleIndexLike3d = (0, 0, 0)
) -> SingleStackedIndexLike3d:
    ...


@overload
def wrap_index_around_origin_x01(
    basis: _B3d0Inv,
    idx: ArrayStackedIndexLike3d[_S0Inv],
    origin_idx: SingleIndexLike3d = (0, 0, 0),
) -> ArrayStackedIndexLike3d[_S0Inv]:
    ...


@overload
def wrap_index_around_origin_x01(
    basis: _B3d0Inv,
    idx: ArrayFlatIndexLike[_S0Inv],
    origin_idx: SingleIndexLike3d = (0, 0, 0),
) -> ArrayStackedIndexLike3d[_S0Inv]:
    ...


def wrap_index_around_origin_x01(
    basis: _B3d0Inv,
    idx: StackedIndexLike3d | FlatIndexLike,
    origin_idx: SingleIndexLike3d = (0, 0, 0),
) -> StackedIndexLike3d:
    """
    Given an index or list of indexes in stacked form, find the equivalent index closest to the point origin_idx.

    Parameters
    ----------
    basis : _B3d0Inv
    idx : StackedIndexLike | FlatIndexLike
    origin_idx : StackedIndexLike | FlatIndexLike, optional
        origin to wrap around, by default (0, 0, 0)

    Returns
    -------
    StackedIndexLike
    """
    util = Basis3dUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    origin_idx = (
        origin_idx
        if isinstance(origin_idx, tuple)
        else util.get_stacked_index(origin_idx)
    )
    (n0, n1, _) = util.shape
    return (  # type: ignore[return-value]
        _wrap_distance(idx[0] - origin_idx[0], n0) + origin_idx[0],
        _wrap_distance(idx[1] - origin_idx[1], n1) + origin_idx[1],
        idx[2],
    )


def calculate_distances_along_path(
    basis: _B0Inv,
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """
    calculate cumulative distances along the given path.

    Parameters
    ----------
    basis : _B0Inv
        basis which the path is through
    path : np.ndarray[tuple[int, int], np.dtype[np.int_]]
        path through the basis, _ND by int points
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.int_]]
    """
    out = path[:, :-1] - path[:, 1:]
    if wrap_distances:
        util = BasisUtil(basis)
        return np.array(  # type: ignore[no-any-return]
            [_wrap_distance(d, n) for (d, n) in zip(out, util.shape, strict=True)]
        )

    return out  # type:ignore[no-any-return]


def calculate_cumulative_x_distances_along_path(
    basis: _B0Inv,
    path: np.ndarray[tuple[int, int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of indexes in the basis, _ND by int points
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    distances = calculate_distances_along_path(
        basis, path, wrap_distances=wrap_distances
    )

    util = BasisUtil(basis)
    x_distances = np.linalg.norm(
        np.tensordot(util.fundamental_dx, distances, axes=(0, 0)), axis=0
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


def calculate_cumulative_k_distances_along_path(
    basis: _B3d0Inv,
    path: np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]],
    *,
    wrap_distances: bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
    """
    calculate the cumulative distances along the given path in the given basis.

    Parameters
    ----------
    basis : _B3d0Inv
        basis which the path is through
    path : np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]
        path, as a list of indexes in the basis
    wrap_distances : bool, optional
        wrap the distances into the first unit cell, by default False

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.float_]]
    """
    (d0, d1, d2) = calculate_distances_along_path(
        basis, path, wrap_distances=wrap_distances  # type: ignore[arg-type]
    )
    util = Basis3dUtil(basis)
    x_distances = np.linalg.norm(
        d0[np.newaxis, :] * util.dk0[:, np.newaxis]
        + d1[np.newaxis, :] * util.dk1[:, np.newaxis]
        + d2[np.newaxis, :] * util.dk2[:, np.newaxis],
        axis=0,
    )
    cum_distances = np.cumsum(x_distances)
    # Add back initial distance
    return np.insert(cum_distances, 0, 0)  # type: ignore[no-any-return]


@overload
def get_x01_mirrored_index(
    basis: _B3d0Inv, idx: SingleStackedIndexLike3d
) -> SingleStackedIndexLike3d:
    ...


@overload
def get_x01_mirrored_index(basis: _B3d0Inv, idx: SingleFlatIndexLike) -> np.int_:
    ...


@overload
def get_x01_mirrored_index(
    basis: _B3d0Inv, idx: ArrayStackedIndexLike3d[_S0Inv]
) -> ArrayStackedIndexLike3d[_S0Inv]:
    ...


@overload
def get_x01_mirrored_index(
    basis: _B3d0Inv, idx: ArrayFlatIndexLike[_S0Inv]
) -> ArrayFlatIndexLike[_S0Inv]:
    ...


def get_x01_mirrored_index(basis: _B3d0Inv, idx: IndexLike3d) -> IndexLike3d:
    """
    Mirror the coordinate idx about x0=x1.

    Parameters
    ----------
    basis : _B3d0Inv
        the basis to mirror in
    idx : tuple[int, int, int] | int
        The index to mirror

    Returns
    -------
    tuple[int, int, int] | int
        The mirrored index
    """
    util = Basis3dUtil(basis)
    idx = idx if isinstance(idx, tuple) else util.get_stacked_index(idx)
    mirrored: StackedIndexLike3d = (idx[1], idx[0], idx[2])  # type: ignore[assignment]
    return mirrored if isinstance(idx, tuple) else util.get_flat_index(mirrored)


def get_single_point_basis(
    basis: _B3d0Inv,
) -> Basis3d[
    FundamentalPositionAxis3d[Literal[1]],
    FundamentalPositionAxis3d[Literal[1]],
    FundamentalPositionAxis3d[Literal[1]],
]:
    """
    Get the basis with a single point in position space.

    Parameters
    ----------
    basis : _B3d0Inv
        initial basis
    _type : Literal[&quot;position&quot;, &quot;momentum&quot;]
        type of the final basis

    Returns
    -------
    _SPB|_SMB
        the single point basis in either position or momentum basis
    """
    return tuple(FundamentalPositionAxis3d(b.delta_x, 1) for b in basis)  # type: ignore[return-value]
