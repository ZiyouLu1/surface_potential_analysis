from typing import Generic, Literal, TypeVar

import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis import (
    BasisUtil,
    ExplicitBasis,
    MomentumBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis_config import BasisConfig, BasisConfigUtil
from surface_potential_analysis.hamiltonian import HamiltonianWithBasis
from surface_potential_analysis.potential import Potential

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


class PotentialSizeError(Exception):
    """Error thrown when the potential is too small."""

    def __init__(self, axis: int, required: int, actual: int) -> None:
        super().__init__(
            f"Potential does not have enough resolution in x{axis} direction"
            f"required {required} actual {actual}"
        )


class _SurfaceHamiltonianUtil(Generic[_L0, _L1, _L2, _L3, _L4, _L5]):
    _potential: Potential[_L0, _L1, _L2]

    _basis: BasisConfig[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, MomentumBasis[_L2]],
    ]
    _mass: float

    def __init__(
        self,
        potential: Potential[_L0, _L1, _L2],
        basis: BasisConfig[
            TruncatedBasis[_L3, MomentumBasis[_L0]],
            TruncatedBasis[_L4, MomentumBasis[_L1]],
            ExplicitBasis[_L5, MomentumBasis[_L2]],
        ],
        mass: float,
    ) -> None:
        self._potential = potential
        self._basis = basis
        self._mass = mass
        if 2 * (self._basis[0]["n"] - 1) > self._potential["basis"][0]["n"]:
            raise PotentialSizeError(
                0, 2 * (self._basis[0]["n"] - 1), self._potential["basis"][0]["n"]
            )

        if 2 * (self._basis[1]["n"] - 1) > self._potential["basis"][1]["n"]:
            raise PotentialSizeError(
                1, 2 * (self._basis[1]["n"] - 1), self._potential["basis"][1]["n"]
            )

    @property
    def points(self) -> np.ndarray[tuple[_L0, _L1, _L2], np.dtype[np.float_]]:
        return self._potential["points"]

    @property
    def nx(self) -> int:
        return self.points.shape[0]  # type:ignore[no-any-return]

    @property
    def ny(self) -> int:
        return self.points.shape[1]  # type:ignore[no-any-return]

    @property
    def nz(self) -> int:
        return self.points.shape[2]  # type:ignore[no-any-return]

    @property
    def dz(self) -> float:
        util = BasisUtil(self._potential["basis"][2])
        return np.linalg.norm(util.fundamental_dx)  # type:ignore[return-value]

    def hamiltonian(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWithBasis[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, MomentumBasis[_L2]],
    ]:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(bloch_phase))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies

        return {"array": energies, "basis": self._basis}

    def _calculate_diagonal_energy_fundamental_x2(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        util = BasisConfigUtil(self._basis)

        kx0_coords, kx1_coords, kx2_fundamental = np.meshgrid(
            util.x0_basis.nk_points,  # type: ignore[misc]
            util.x1_basis.nk_points,  # type: ignore[misc]
            util.x2_basis.fundamental_nk_points,  # type: ignore[misc]
            indexing="ij",
        )

        dkx0 = util.dk0
        dkx1 = util.dk1
        mass = self._mass

        k0_points = dkx0[0] * kx0_coords + dkx1[0] * kx1_coords + bloch_phase[0]
        x0_energy = (hbar * k0_points) ** 2 / (2 * mass)
        k1_points = dkx0[1] * kx0_coords + dkx1[1] * kx1_coords + bloch_phase[1]
        x1_energy = (hbar * k1_points) ** 2 / (2 * mass)
        k2_points = dkx0[2] * kx0_coords + dkx1[2] * kx1_coords + bloch_phase[2]
        x2_energy = (hbar * k2_points) ** 2 / (2 * mass)
        return x0_energy + x1_energy + x2_energy  # type: ignore[no-any-return]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        return np.fft.ifft2(self._potential["points"], axes=(0, 1, 2), norm="ortho")  # type: ignore[no-any-return]


def total_surface_hamiltonian(
    potential: Potential[_L0, _L1, _L2],
    bloch_phase: np.ndarray[tuple[Literal[2]], np.dtype[np.float_]],
    basis: BasisConfig[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, MomentumBasis[_L2]],
    ],
    mass: float,
) -> HamiltonianWithBasis[
    TruncatedBasis[_L3, MomentumBasis[_L0]],
    TruncatedBasis[_L4, MomentumBasis[_L1]],
    ExplicitBasis[_L5, MomentumBasis[_L2]],
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
