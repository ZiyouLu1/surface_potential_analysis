from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis import (
    BasisUtil,
)
from surface_potential_analysis.basis_config.util import BasisConfigUtil

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        ExplicitBasis,
        MomentumBasis,
    )
    from surface_potential_analysis.basis_config.basis_config import (
        BasisConfig,
    )
    from surface_potential_analysis.hamiltonian import HamiltonianWithBasis
    from surface_potential_analysis.potential import Potential

_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)
_N2Inv = TypeVar("_N2Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)


class PotentialSizeError(Exception):
    """Error thrown when the potential is too small."""

    def __init__(self, axis: int, required: int, actual: int) -> None:
        super().__init__(
            f"Potential does not have enough resolution in x{axis} direction"
            f"required {required} actual {actual}"
        )


class _SurfaceHamiltonianUtil(
    Generic[_N0Inv, _N1Inv, _N2Inv, _NF0Inv, _NF1Inv, _NF2Inv]
):
    _potential: Potential[_NF0Inv, _NF1Inv, _NF2Inv]

    _basis: BasisConfig[
        MomentumBasis[_NF0Inv, _N0Inv],
        MomentumBasis[_NF1Inv, _N1Inv],
        ExplicitBasis[_NF2Inv, _N2Inv],
    ]
    _mass: float

    def __init__(
        self,
        potential: Potential[_NF0Inv, _NF1Inv, _NF2Inv],
        basis: BasisConfig[
            MomentumBasis[_NF0Inv, _N0Inv],
            MomentumBasis[_NF1Inv, _N1Inv],
            ExplicitBasis[_NF2Inv, _N2Inv],
        ],
        mass: float,
    ) -> None:
        self._potential = potential
        self._basis = basis
        self._mass = mass
        if 2 * (self._basis[0].n - 1) > self._potential["basis"][0].n:
            raise PotentialSizeError(
                0, 2 * (self._basis[0].n - 1), self._potential["basis"][0].n
            )

        if 2 * (self._basis[1].n - 1) > self._potential["basis"][1].n:
            raise PotentialSizeError(
                1, 2 * (self._basis[1].n - 1), self._potential["basis"][1].n
            )

    @property
    def points(
        self,
    ) -> np.ndarray[tuple[_NF0Inv, _NF1Inv, _NF2Inv], np.dtype[np.float_]]:
        return self._potential["points"]

    @property
    def nx(self) -> int:
        return self.points.shape[0]  # type: ignore[no-any-return]

    @property
    def ny(self) -> int:
        return self.points.shape[1]  # type: ignore[no-any-return]

    @property
    def nz(self) -> int:
        return self.points.shape[2]  # type: ignore[no-any-return]

    @property
    def dz(self) -> float:
        util = BasisUtil(self._potential["basis"][2])
        return np.linalg.norm(util.fundamental_dx)  # type: ignore[return-value]

    def hamiltonian(
        self, _bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWithBasis[
        MomentumBasis[_NF0Inv, _N0Inv],
        MomentumBasis[_NF1Inv, _N1Inv],
        ExplicitBasis[_NF2Inv, _N2Inv],
    ]:
        raise NotImplementedError

    def _calculate_diagonal_energy_fundamental_x2(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        util = BasisConfigUtil(self._basis)

        k0_coords, k1_coords, k2_fundamental = np.meshgrid(
            util.x0_basis.nk_points,  # type: ignore[misc]
            util.x1_basis.nk_points,  # type: ignore[misc]
            util.x2_basis.fundamental_nk_points,  # type: ignore[misc]
            indexing="ij",
        )

        dk0 = util.dk0
        dk1 = util.dk1
        mass = self._mass

        k0_points = dk0[0] * k0_coords + dk1[0] * k1_coords + bloch_phase[0]
        x0_energy = (hbar * k0_points) ** 2 / (2 * mass)
        k1_points = dk0[1] * k0_coords + dk1[1] * k1_coords + bloch_phase[1]
        x1_energy = (hbar * k1_points) ** 2 / (2 * mass)
        k2_points = dk0[2] * k0_coords + dk1[2] * k1_coords + bloch_phase[2]
        x2_energy = (hbar * k2_points) ** 2 / (2 * mass)
        return x0_energy + x1_energy + x2_energy  # type: ignore[no-any-return]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        return np.fft.ifft2(self._potential["points"], axes=(0, 1, 2), norm="ortho")  # type: ignore[no-any-return]


def total_surface_hamiltonian(
    potential: Potential[_NF0Inv, _NF1Inv, _NF2Inv],
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    basis: BasisConfig[
        MomentumBasis[_NF0Inv, _N0Inv],
        MomentumBasis[_NF1Inv, _N1Inv],
        ExplicitBasis[_NF2Inv, _N2Inv],
    ],
    mass: float,
) -> HamiltonianWithBasis[
    MomentumBasis[_NF0Inv, _N0Inv],
    MomentumBasis[_NF1Inv, _N1Inv],
    ExplicitBasis[_NF2Inv, _N2Inv],
]:
    """
    Calculate a hamiltonian using the given basis.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    bloch_phase : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    basis : BasisConfig[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    mass : float

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    """
    util = _SurfaceHamiltonianUtil(potential, basis, mass)
    return util.hamiltonian(bloch_phase)
