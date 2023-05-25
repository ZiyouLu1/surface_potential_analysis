from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
    FundamentalPositionBasis,
    MomentumBasis,
)
from surface_potential_analysis.basis_config.sho_basis import (
    SHOBasisConfig,
    infinate_sho_basis_from_config,
)
from surface_potential_analysis.basis_config.util import BasisConfigUtil
from surface_potential_analysis.eigenstate.conversion import (
    convert_eigenstate_to_basis,
)

from .utils import get_random_explicit_basis

if TYPE_CHECKING:
    from surface_potential_analysis.eigenstate.eigenstate import (
        EigenstateWithBasis,
    )

_rng = np.random.default_rng()


def _get_random_sho_eigenstate(
    resolution: tuple[int, int, int], fundamental_resolution: tuple[int, int, int]
) -> EigenstateWithBasis[
    MomentumBasis[Any, Any],
    MomentumBasis[Any, Any],
    ExplicitBasis[int, Any],
]:
    vector = np.array(_rng.random(np.prod(resolution)), dtype=complex)
    vector /= np.linalg.norm(vector)

    x2_basis = x2_basis = ExplicitBasis(
        np.array([0, 0, 20]),
        get_random_explicit_basis(
            fundamental_n=fundamental_resolution[2], n=resolution[2]
        ).vectors,
    )
    return {
        "basis": (
            MomentumBasis(
                np.array([1, 0, 0]), resolution[0], fundamental_resolution[0]
            ),
            MomentumBasis(
                np.array([0, 1, 0]), resolution[1], fundamental_resolution[1]
            ),
            x2_basis,
        ),
        "vector": vector,
    }


class EigenstateConversionTest(unittest.TestCase):
    def test_convert_sho_eigenstate_rust_simple(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2], config, resolution[2]
            ),
        )

        util = BasisConfigUtil(eigenstate["basis"])

        for i in range(resolution[2]):
            vector = np.zeros_like(eigenstate["vector"])
            vector[np.ravel_multi_index((0, 0, i), resolution)] = 1
            eigenstate["vector"] = vector

            points = util.fundamental_x_points + config["x_origin"][:, np.newaxis]

            actual = hamiltonian_generator.get_eigenstate_wavefunction(
                resolution,
                (util.delta_x0.item(0), 0),
                (util.delta_x1.item(0), util.delta_x1.item(1)),
                config["mass"],
                config["sho_omega"],
                0,
                0,
                eigenstate["vector"].tolist(),
                points.T.tolist(),
            )

            basis = (
                FundamentalPositionBasis(util.delta_x0, util.fundamental_n0),  # type: ignore[misc]
                FundamentalPositionBasis(util.delta_x1, util.fundamental_n1),  # type: ignore[misc]
                FundamentalPositionBasis(util.delta_x2, util.fundamental_n2),  # type: ignore[misc]
            )
            expected = convert_eigenstate_to_basis(eigenstate, basis)
            np.testing.assert_allclose(
                expected["vector"], np.array(actual) / np.linalg.norm(actual)
            )

    def test_convert_sho_eigenstate_rust(self) -> None:
        resolution = (5, 6, 9)
        eigenstate = _get_random_sho_eigenstate(resolution, (10, 10, 100))
        config: SHOBasisConfig = {
            "mass": hbar**2,
            "sho_omega": 1 / hbar,
            "x_origin": np.array([0, 0, -10]),
        }

        eigenstate["basis"] = (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            infinate_sho_basis_from_config(
                eigenstate["basis"][2], config, resolution[2]
            ),
        )

        util = BasisConfigUtil(eigenstate["basis"])

        points = util.fundamental_x_points + config["x_origin"][:, np.newaxis]
        actual = hamiltonian_generator.get_eigenstate_wavefunction(
            resolution,
            (util.delta_x0.item(0), 0),
            (util.delta_x1.item(0), util.delta_x1.item(1)),
            config["mass"],
            config["sho_omega"],
            0,
            0,
            eigenstate["vector"].tolist(),
            points.T.tolist(),
        )

        basis = (
            FundamentalPositionBasis(util.delta_x0, util.fundamental_n0),  # type: ignore[misc]
            FundamentalPositionBasis(util.delta_x1, util.fundamental_n1),  # type: ignore[misc]
            FundamentalPositionBasis(util.delta_x2, util.fundamental_n2),  # type: ignore[misc]
        )
        expected = convert_eigenstate_to_basis(eigenstate, basis)
        np.testing.assert_allclose(
            expected["vector"], np.array(actual) / np.linalg.norm(actual)
        )
