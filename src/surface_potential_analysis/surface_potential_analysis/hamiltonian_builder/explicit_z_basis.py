from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.axis.axis import (
    ExplicitAxis,
    ExplicitAxis3d,
    FundamentalPositionAxis1d,
    FundamentalTransformedPositionAxis3d,
    TransformedPositionAxis,
    TransformedPositionAxis2d,
    TransformedPositionAxis3d,
)
from surface_potential_analysis.basis.potential_basis import (
    get_potential_basis_config_eigenstates,
)
from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.basis.potential_basis import PotentialBasisConfig
    from surface_potential_analysis.operator import SingleBasisOperator
    from surface_potential_analysis.potential import (
        FundamentalPositionBasisPotential3d,
    )
    from surface_potential_analysis.state_vector.eigenstate_calculation import (
        EigenvectorList,
    )

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
    _potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv]

    _resolution: tuple[_N0Inv, _N1Inv]
    _config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_NF2Inv]], _N2Inv]
    state_vectors_z: EigenvectorList[tuple[FundamentalPositionAxis1d[_NF2Inv]], _N2Inv]

    def __init__(
        self,
        potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv],
        resolution: tuple[_N0Inv, _N1Inv],
        config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_NF2Inv]], _N2Inv],
    ) -> None:
        self._potential = potential
        self._resolution = resolution
        self._config = config
        self.state_vectors_z = get_potential_basis_config_eigenstates(self._config)
        if 2 * (self.basis[0].n - 1) > self._potential["basis"][0].n:
            raise PotentialSizeError(
                0, 2 * (self.basis[0].n - 1), self._potential["basis"][0].n
            )

        if 2 * (self.basis[1].n - 1) > self._potential["basis"][1].n:
            raise PotentialSizeError(
                1, 2 * (self.basis[1].n - 1), self._potential["basis"][1].n
            )

    @property
    def basis(
        self,
    ) -> tuple[
        TransformedPositionAxis[_NF0Inv, _N0Inv, Literal[3]],
        TransformedPositionAxis[_NF1Inv, _N1Inv, Literal[3]],
        ExplicitAxis[_NF2Inv, _N2Inv, Literal[3]],
    ]:
        return (
            TransformedPositionAxis3d(
                self._potential["basis"][0].delta_x,
                self._resolution[0],
                self._potential["basis"][0].n,
            ),
            TransformedPositionAxis3d(
                self._potential["basis"][1].delta_x,
                self._resolution[1],
                self._potential["basis"][1].n,
            ),
            ExplicitAxis3d(
                self._potential["basis"][2].delta_x,
                self.state_vectors_z["vectors"],  # type: ignore[arg-type]
            ),
        )

    @property
    def points(
        self,
    ) -> np.ndarray[tuple[_NF0Inv, _NF1Inv, _NF2Inv], np.dtype[np.complex_]]:
        return self._potential["vector"].reshape(  # type: ignore[no-any-return]
            AxisWithLengthBasisUtil(self._potential["basis"]).shape
        )

    def hamiltonian(
        self, bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> SingleBasisOperator[
        tuple[
            TransformedPositionAxis[_NF0Inv, _N0Inv, Literal[3]],
            TransformedPositionAxis[_NF1Inv, _N1Inv, Literal[3]],
            ExplicitAxis[_NF2Inv, _N2Inv, Literal[3]],
        ]
    ]:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(bloch_fraction))
        other_energies = self._calculate_off_diagonal_energies()

        energies = diagonal_energies + other_energies

        return {"array": energies, "basis": self.basis, "dual_basis": self.basis}

    def _calculate_off_diagonal_energies(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
        return np.array(  # type: ignore[no-any-return]
            hamiltonian_generator.calculate_off_diagonal_energies2(
                self.get_ft_potential().tolist(),
                self.basis[2].vectors.tolist(),
                AxisWithLengthBasisUtil(self.basis).shape,  # type: ignore[arg-type]
            )
        )

    def _calculate_diagonal_energy(
        self, bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        # Note we ignore bloch_fraction in x2
        xy_basis = (
            TransformedPositionAxis2d(
                self.basis[0].delta_x[:2], self.basis[0].n, self.basis[0].fundamental_n
            ),
            TransformedPositionAxis2d(
                self.basis[1].delta_x[:2], self.basis[1].n, self.basis[1].fundamental_n
            ),
        )
        util = AxisWithLengthBasisUtil(xy_basis)

        bloch_phase = np.tensordot(util.dk, bloch_fraction[:2], axes=(0, 0))
        k_points = util.k_points + bloch_phase[:, np.newaxis]
        xy_energy = np.sum(
            np.square(hbar * k_points) / (2 * self._config["mass"]),
            axis=0,
            dtype=np.complex_,
        )
        z_energy = self.state_vectors_z["eigenvalues"]
        return (xy_energy[:, np.newaxis] + z_energy[np.newaxis, :]).ravel()  # type: ignore[no-any-return]

    @property
    def subtracted_points(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float_]]:
        return np.subtract(self.points, self._config["potential"]["vector"][np.newaxis, np.newaxis, :])  # type: ignore[no-any-return]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        return np.fft.ifft2(self.subtracted_points, axes=(0, 1))  # type: ignore[no-any-return]


@timed
def total_surface_hamiltonian(
    potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_N0Inv, _N1Inv],
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_NF2Inv]], _N2Inv],
) -> SingleBasisOperator[
    tuple[
        TransformedPositionAxis[_NF0Inv, _N0Inv, Literal[3]],
        TransformedPositionAxis[_NF1Inv, _N1Inv, Literal[3]],
        ExplicitAxis[_NF2Inv, _N2Inv, Literal[3]],
    ]
]:
    """
    Calculate a hamiltonian using the given basis.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    basis : Basis3d[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    mass : float

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    """
    util = _SurfaceHamiltonianUtil(potential, resolution, config)
    return util.hamiltonian(bloch_fraction)


def total_surface_hamiltonian_as_fundamental(
    potential: FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    resolution: tuple[_N0Inv, _N1Inv],
    config: PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_NF2Inv]], _N2Inv],
) -> SingleBasisOperator[
    tuple[
        TransformedPositionAxis[_N0Inv, _N0Inv, Literal[3]],
        TransformedPositionAxis[_N1Inv, _N1Inv, Literal[3]],
        ExplicitAxis[_NF2Inv, _N2Inv, Literal[3]],
    ]
]:
    """
    calculate the hamiltonian, and ignore the true shape in the xy direction.

    Parameters
    ----------
    potential : FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv]
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    resolution : tuple[_N0Inv, _N1Inv]
    config : PotentialBasisConfig[tuple[FundamentalPositionAxis1d[_NF2Inv]], _N2Inv]

    Returns
    -------
    SingleBasisOperator[
        tuple[ MomentumAxis3d[_NF0Inv, _NF0Inv], MomentumAxis3d[_NF1Inv, _NF1Inv], ExplicitAxis3d[_NF2Inv, _N2Inv], ]
    """
    hamiltonian = total_surface_hamiltonian(
        potential, bloch_fraction, resolution, config
    )
    return {
        "basis": (
            FundamentalTransformedPositionAxis3d(
                hamiltonian["basis"][0].delta_x, hamiltonian["basis"][0].n
            ),
            FundamentalTransformedPositionAxis3d(
                hamiltonian["basis"][1].delta_x, hamiltonian["basis"][1].n
            ),
            hamiltonian["basis"][2],
        ),
        "array": hamiltonian["array"],
        "dual_basis": (
            FundamentalTransformedPositionAxis3d(
                hamiltonian["dual_basis"][0].delta_x, hamiltonian["dual_basis"][0].n
            ),
            FundamentalTransformedPositionAxis3d(
                hamiltonian["dual_basis"][1].delta_x, hamiltonian["dual_basis"][1].n
            ),
            hamiltonian["dual_basis"][2],
        ),
    }
