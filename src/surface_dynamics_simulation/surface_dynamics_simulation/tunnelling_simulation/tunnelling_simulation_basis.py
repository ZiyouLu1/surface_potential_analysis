from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from surface_potential_analysis.axis.axis_like import AxisLike3d, AxisVector3d

if TYPE_CHECKING:
    import numpy as np

_NF0Inv = TypeVar("_NF0Inv", bound=int)
_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)


# ruff: noqa: D102
class TunnellingSimulationBasis(AxisLike3d[_NF0Inv, _N0Inv]):
    """
    Represents the basis of a tunnelling simulation.

    This representation makes use of the
    symmetry of the basis vectors on each site to reduce memory usage.
    """

    def __init__(
        self,
        delta_x: AxisVector3d,
        single_site_vectors: np.ndarray[tuple[_N1Inv, _NF0Inv], np.dtype[np.complex_]],
        n_sites: int,
    ) -> None:
        self._delta_x = delta_x
        self._single_site_vectors = single_site_vectors
        self._n_sites = n_sites

    @property
    def vectors(self) -> np.ndarray[tuple[_N0Inv, _NF0Inv], np.dtype[np.complex_]]:
        raise NotImplementedError

    @property
    def delta_x(self) -> AxisVector3d:
        return self._delta_x

    @property
    def n_bands(self) -> int:
        return self._single_site_vectors.shape[0]  # type: ignore[no-any-return]

    @property
    def n(self) -> _N0Inv:
        return self.n_bands * self._n_sites  # type: ignore[return-value]

    @property
    def fundamental_n(self) -> _NF0Inv:
        return self._single_site_vectors.shape[1]  # type: ignore[no-any-return]
