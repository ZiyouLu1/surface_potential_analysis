from functools import cached_property
from typing import Generic, Literal, TypeVar

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis import (
    BasisUtil,
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.basis_config.sho_basis import (
    SHOBasisConfig,
    calculate_x_distances,
    infinate_sho_basis_from_config,
)
from surface_potential_analysis.hamiltonian import HamiltonianWithBasis
from surface_potential_analysis.potential import Potential

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


class _SurfaceHamiltonianUtil(Generic[_L0, _L1, _L2, _L3, _L4, _L5]):
    _potential: Potential[_L0, _L1, _L2]

    _config: SHOBasisConfig

    _resolution: tuple[_L3, _L4, _L5]

    def __init__(
        self,
        potential: Potential[_L0, _L1, _L2],
        config: SHOBasisConfig,
        resolution: tuple[_L3, _L4, _L5],
    ) -> None:
        self._potential = potential
        self._config = config
        self._resolution = resolution
        if 2 * (self._resolution[0] - 1) > self._potential["basis"][0]["n"]:
            raise AssertionError(  # noqa: TRY003
                "Not have enough resolution in x0"  # noqa: EM101
            )

        if 2 * (self._resolution[1] - 1) > self._potential["basis"][1]["n"]:
            raise AssertionError(  # noqa: TRY003
                "Not have enough resolution in x1"  # noqa: EM101
            )

    @property
    def points(self) -> np.ndarray[tuple[_L0, _L1, _L2], np.dtype[np.float_]]:
        return self._potential["points"]

    @property
    def z_offset(self) -> float:
        return self._config["x_origin"][2]  # type:ignore[return-value]

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

    @property
    def z_distances(self) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        return calculate_x_distances(
            self._potential["basis"][2], self._config["x_origin"]  # type: ignore[arg-type]
        )

    @property
    def basis(
        self,
    ) -> BasisConfig[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, PositionBasis[_L2]],
    ]:
        return (
            {
                "_type": "truncated",
                "n": self._resolution[0],
                "parent": {
                    "_type": "momentum",
                    "delta_x": self._potential["basis"][0]["delta_x"],
                    "n": self._potential["basis"][0]["n"],
                },
            },
            {
                "_type": "truncated",
                "n": self._resolution[1],
                "parent": {
                    "_type": "momentum",
                    "delta_x": self._potential["basis"][1]["delta_x"],
                    "n": self._potential["basis"][1]["n"],
                },
            },
            infinate_sho_basis_from_config(
                {
                    "_type": "position",
                    "delta_x": self._potential["basis"][2]["delta_x"],
                    "n": self._potential["basis"][2]["n"],
                },
                self._config,
                self._resolution[2],
            ),
        )

    def hamiltonian(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWithBasis[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, PositionBasis[_L2]],
    ]:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(bloch_phase))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies

        return {"array": energies, "basis": self.basis}

    @cached_property
    def eigenstate_indexes(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        util = BasisConfigUtil(self.basis)

        x0t, x1t, zt = np.meshgrid(
            util.x0_basis.nk_points,  # type: ignore[misc]
            util.x1_basis.nk_points,  # type: ignore[misc]
            util.x2_basis.nx_points,  # type: ignore[misc]
            indexing="ij",
        )
        return np.array([x0t.ravel(), x1t.ravel(), zt.ravel()])  # type: ignore[no-any-return]

    def _calculate_diagonal_energy(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        k0_coords, k1_coords, nkz_coords = self.eigenstate_indexes

        util = BasisConfigUtil(self.basis)

        dk0 = util.dk0
        dk1 = util.dk1
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]

        kx_points = dk0[0] * k0_coords + dk1[0] * k1_coords + bloch_phase[0]
        x_energy = (hbar * kx_points) ** 2 / (2 * mass)
        ky_points = dk0[1] * k0_coords + dk1[1] * k1_coords + bloch_phase[1]
        y_energy = (hbar * ky_points) ** 2 / (2 * mass)
        z_energy = (hbar * sho_omega) * (nkz_coords + 0.5)
        return x_energy + y_energy + z_energy  # type: ignore[no-any-return]

    def get_sho_potential(self) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]
        return 0.5 * mass * sho_omega**2 * np.square(self.z_distances)  # type: ignore[no-any-return]

    def get_sho_subtracted_points(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float_]]:
        return np.subtract(self.points, self.get_sho_potential())  # type: ignore[no-any-return]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        subtracted_potential = self.get_sho_subtracted_points()
        return np.fft.ifft2(subtracted_potential, axes=(0, 1))  # type: ignore[no-any-return]

    def _calculate_off_diagonal_energies_fast(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]
        return np.array(  # type: ignore[no-any-return]
            hamiltonian_generator.calculate_off_diagonal_energies(
                self.get_ft_potential().tolist(),
                self._resolution,
                self.dz,
                mass,
                sho_omega,
                self.z_offset,
            )
        )

    def _calculate_off_diagonal_entry(
        self, nz1: int, nz2: int, ndk0: int, ndk1: int
    ) -> float:
        """Calculate the off diagonal energy using the 'folded' points ndkx, ndky."""
        ft_pot_points = self.get_ft_potential()[ndk0, ndk1]
        hermite1 = self.basis[2]["vectors"][nz1]
        hermite2 = self.basis[2]["vectors"][nz2]

        fourier_transform = float(np.sum(hermite1 * hermite2 * ft_pot_points))

        return self.dz * fourier_transform

    def _calculate_off_diagonal_energies(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
        n_coordinates = len(self.eigenstate_indexes.T)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for index1, [nk0_0, nk1_0, nz1] in enumerate(self.eigenstate_indexes.T):
            for index2, [nk0_1, nk1_1, nz2] in enumerate(self.eigenstate_indexes.T):
                # Number of jumps in units of dkx for this matrix element

                # As k-> k+ Nx * dkx the ft potential is left unchanged
                # Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
                # Where Nx is the number of x points

                # In reality we want to make sure ndkx < Nx (ie we should
                # make sure we generate enough points in the interpolation)
                ndk0 = (nk0_1 - nk0_0) % self.nx
                ndk1 = (nk1_1 - nk1_0) % self.ny

                hamiltonian[index1, index2] = self._calculate_off_diagonal_entry(
                    nz1, nz2, ndk0, ndk1
                )

        return hamiltonian  # type: ignore[no-any-return]


def total_surface_hamiltonian(
    potential: Potential[_L0, _L1, _L2],
    config: SHOBasisConfig,
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_L3, _L4, _L5],
) -> HamiltonianWithBasis[
    TruncatedBasis[_L3, MomentumBasis[_L0]],
    TruncatedBasis[_L4, MomentumBasis[_L1]],
    ExplicitBasis[_L5, PositionBasis[_L2]],
]:
    """
    Calculate a hamiltonian using the infinite sho basis.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    config : SHOBasisConfig
    bloch_phase : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    resolution : tuple[_L3, _L4, _L5]

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, PositionBasis[_L2]]]
    """
    util = _SurfaceHamiltonianUtil(potential, config, resolution)
    return util.hamiltonian(bloch_phase)
