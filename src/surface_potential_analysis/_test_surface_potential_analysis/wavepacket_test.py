from __future__ import annotations

import unittest
from typing import TYPE_CHECKING, Any

import numpy as np

from surface_potential_analysis.basis.build import (
    fundamental_basis_from_shape,
    momentum_basis_3d_from_resolution,
    position_basis_3d_from_shape,
)
from surface_potential_analysis.wavepacket.eigenstate_conversion import (
    furl_eigenstate,
    unfurl_wavepacket,
)
from surface_potential_analysis.wavepacket.localization._tight_binding import (
    _get_global_phases,
)
from surface_potential_analysis.wavepacket.localization._wannier90 import (
    _parse_nnkpts_file,  # cSpell:disable-line
    _parse_u_mat_file,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_sample_fractions,
)

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.wavepacket import (
        MomentumBasisWavepacket3d,
        PositionBasisWavepacket3d,
    )

rng = np.random.default_rng()


class WavepacketTest(unittest.TestCase):
    def test_get_global_phases(self) -> None:
        ns0 = rng.integers(1, 10)
        ns1 = rng.integers(1, 10)
        resolution = (
            rng.integers(2, 10),
            rng.integers(1, 10),
            rng.integers(1, 10),
        )
        wavepacket: PositionBasisWavepacket3d[Any, Any, Any, Any, Any] = {
            "basis": position_basis_3d_from_shape(resolution),
            "vectors": np.zeros((ns0 * ns1, np.prod(resolution))),
            "eigenvalues": np.zeros(ns0 * ns1),
            "list_basis": fundamental_basis_from_shape((ns0, ns1, 1)),  # type: ignore[typeddict-item]
        }

        idx = rng.integers(0, np.prod(resolution).item())
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual.shape, (ns0 * ns1,))
        np.testing.assert_equal(actual[0], 0)

        idx = 0
        actual = _get_global_phases(wavepacket, idx)
        np.testing.assert_array_equal(actual, np.zeros_like(actual))

        idx_array = rng.integers(0, np.prod(resolution).item(), size=(10, 10, 11))
        actual_large = _get_global_phases(wavepacket, idx_array)
        np.testing.assert_array_equal(actual_large.shape, (ns0 * ns1, *idx_array.shape))
        np.testing.assert_equal(actual_large[0], 0)
        np.testing.assert_equal(actual_large[:, idx_array == 0], 0)

    def test_unfurl_wavepacket(self) -> None:
        wavepacket: MomentumBasisWavepacket3d[int, int, int, int, int] = {
            "basis": momentum_basis_3d_from_resolution((3, 3, 3)),
            "list_basis": fundamental_basis_from_shape((3, 2, 1)),  # type: ignore[typeddict-item]
            "vectors": np.zeros((3, 2, 27)),
            "eigenvalues": np.zeros((3, 2)),
        }
        wavepacket["vectors"][0][0][0] = 1
        wavepacket["vectors"][1][0][0] = 2
        wavepacket["vectors"][2][0][0] = 3
        wavepacket["vectors"][0][1][0] = 4
        wavepacket["vectors"][1][1][0] = 5
        wavepacket["vectors"][2][1][0] = 6

        expected = np.zeros(162)
        expected[np.ravel_multi_index((0, 0, 0), (9, 6, 3))] = 1
        expected[np.ravel_multi_index((1, 0, 0), (9, 6, 3))] = 2
        expected[np.ravel_multi_index((8, 0, 0), (9, 6, 3))] = 3
        expected[np.ravel_multi_index((0, 5, 0), (9, 6, 3))] = 4
        expected[np.ravel_multi_index((1, 5, 0), (9, 6, 3))] = 5
        expected[np.ravel_multi_index((8, 5, 0), (9, 6, 3))] = 6

        eigenstate = unfurl_wavepacket(wavepacket)
        np.testing.assert_array_equal(eigenstate["vector"], expected / np.sqrt(2 * 3))

    def test_furl_eigenstate(self) -> None:
        wavepacket: MomentumBasisWavepacket3d[int, int, int, int, int] = {
            "basis": momentum_basis_3d_from_resolution((3, 3, 3)),
            "vectors": np.array(rng.random((3, 2, 27)), dtype=complex),
            "list_basis": fundamental_basis_from_shape((3, 2, 1)),  # type: ignore[typeddict-item]
            "eigenvalues": np.zeros((3, 2)),
        }
        eigenstate = unfurl_wavepacket(wavepacket)
        actual = furl_eigenstate(eigenstate, (3, 2, 1))

        np.testing.assert_array_almost_equal(wavepacket["vectors"], actual["vectors"])

        np.testing.assert_array_almost_equal(
            wavepacket["basis"][0].delta_x, actual["basis"][0].delta_x
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][1].delta_x, actual["basis"][1].delta_x
        )
        np.testing.assert_array_almost_equal(
            wavepacket["basis"][2].delta_x, actual["basis"][2].delta_x
        )

    def test_get_wavepacket_sample_fractions(self) -> None:
        shape = tuple(rng.integers(1, 10, size=rng.integers(1, 5)))

        actual = get_wavepacket_sample_fractions(shape)
        meshgrid = np.meshgrid(
            *[np.fft.fftfreq(s, 1) for s in shape],
            indexing="ij",
        )
        expected = np.array([x.ravel() for x in meshgrid])
        np.testing.assert_array_almost_equal(expected, actual)
    # ! cSpell:disable
    def test_parse_nnkpts_block(self) -> None:
        block = """
File written on 18Aug2023 at 07:15:55

calc_only_A  :  F

begin real_lattice
   2.4890159   0.0000000   0.0000000
   1.2445079   2.1555510   0.0000000
   0.0000000   0.0000000   2.2000000
end real_lattice

begin recip_lattice
   2.5243653  -1.4574430   0.0000000
   0.0000000   2.9148860   0.0000000
   0.0000000   0.0000000   2.8559933
end recip_lattice

begin projections
     0
end projections

begin auto_projections
     1
     0
end auto_projections

begin nnkpts
   8
     1     2      0   0   0
     1    12      0   0   0
     1    13      0   0   0
     1    14      0   0   0
     1   133      0   0   0
     1   144      0   0   0
     1     1      0   0   1
     1     1      0   0  -1
     2     1      0   0   0
     2     3      0   0   0
     2    14      0   0   0
     2    15      0   0   0
     2   133      0   0   0
     2   134      0   0   0
     2     2      0   0   1
     2     2      0   0  -1
     3     2      0   0   0
     3     4      0   0   0
     3    15      0   0   0
     3    16      0   0   0
     3   134      0   0   0
     3   135      0   0   0
     3     3      0   0   1
     3     3      0   0  -1
   144   133      0   0   0
   144   143      0   0   0
   144   144      0   0   1
   144   144      0   0  -1
end nnkpts
        """
        expected = [
            (1, 2, 0, 0, 0),
            (1, 12, 0, 0, 0),
            (1, 13, 0, 0, 0),
            (1, 14, 0, 0, 0),
            (1, 133, 0, 0, 0),
            (1, 144, 0, 0, 0),
            (1, 1, 0, 0, 1),
            (1, 1, 0, 0, -1),
            (2, 1, 0, 0, 0),
            (2, 3, 0, 0, 0),
            (2, 14, 0, 0, 0),
            (2, 15, 0, 0, 0),
            (2, 133, 0, 0, 0),
            (2, 134, 0, 0, 0),
            (2, 2, 0, 0, 1),
            (2, 2, 0, 0, -1),
            (3, 2, 0, 0, 0),
            (3, 4, 0, 0, 0),
            (3, 15, 0, 0, 0),
            (3, 16, 0, 0, 0),
            (3, 134, 0, 0, 0),
            (3, 135, 0, 0, 0),
            (3, 3, 0, 0, 1),
            (3, 3, 0, 0, -1),
            (144, 133, 0, 0, 0),
            (144, 143, 0, 0, 0),
            (144, 144, 0, 0, 1),
            (144, 144, 0, 0, -1),
        ]
        _, actual = _parse_nnkpts_file(block)
        np.testing.assert_array_equal(expected, actual)

    def test_parse_u_mat_file(self) -> None:
        block = """written on 18Aug2023 at 07:20:28
         144           1           1

  -0.2500000000  -0.2500000000  +0.0000000000
  -0.0724667321  -0.9973708301

  -0.2500000000  -0.1666666667  +0.0000000000
  -0.0553572413  -0.9984666123

  -0.2500000000  -0.0833333333  +0.0000000000
  -0.0573257962  -0.9983555244

  -0.1666666667  +0.0000000000  +0.0000000000
  -0.0427542964  +0.9990856170

  -0.1666666667  +0.0833333333  +0.0000000000
  -0.0090618827  -0.9999589403

  -0.1666666667  +0.1666666667  +0.0000000000
  -0.9892287385  +0.1463779457

  -0.1666666667  +0.2500000000  +0.0000000000
   0.5667187936  -0.8239112871

  -0.1666666667  +0.3333333333  +0.0000000000
   0.6205016526  -0.7842051384

  -0.1666666667  +0.4166666667  +0.0000000000
  -0.7477492359  +0.6639812348

  -0.1666666667  -0.5000000000  +0.0000000000
  -0.2221131667  -0.9750208927

  -0.1666666667  -0.4166666667  +0.0000000000
  -0.1545027834  -0.9879923532

  -0.1666666667  -0.3333333333  +0.0000000000
  -0.1006488398  -0.9949220125

  -0.1666666667  -0.2500000000  +0.0000000000
  -0.0553572413  -0.9984666123

  -0.1666666667  -0.1666666667  +0.0000000000
  -0.0144219837  -0.9998959978

  -0.1666666667  -0.0833333333  +0.0000000000
   0.0214944021  -0.9997689687

  -0.0833333333  +0.0000000000  +0.0000000000
  -0.3288697385  +0.9443752936

  -0.0833333333  +0.0833333333  +0.0000000000
  -0.9892570194  +0.1461866945

  -0.0833333333  +0.1666666667  +0.0000000000
   0.2802951114  +0.9599138766

  -0.0833333333  +0.2500000000  +0.0000000000
  -0.0124033172  -0.9999230759

  -0.0833333333  +0.3333333333  +0.0000000000
  -0.1584949482  +0.9873597882

  -0.0833333333  +0.4166666667  +0.0000000000
  -0.3362508336  +0.9417724656

  -0.0833333333  -0.5000000000  +0.0000000000
  -0.3040457710  -0.9526574249

  -0.0833333333  -0.4166666667  +0.0000000000
  -0.2101846230  -0.9776617126

  -0.0833333333  -0.3333333333  +0.0000000000
  -0.1303232768  -0.9914715546

  -0.0833333333  -0.2500000000  +0.0000000000
  -0.0573257962  -0.9983555244

  -0.0833333333  -0.1666666667  +0.0000000000
   0.0214944021  -0.9997689687

  -0.0833333333  -0.0833333333  +0.0000000000
  -0.1269297494  +0.9919117091
"""
        expected = np.array(
            [
                -0.0724667321 - 0.9973708301j,
                -0.0553572413 - 0.9984666123j,
                -0.0573257962 - 0.9983555244j,
                -0.0427542964 + 0.9990856170j,
                -0.0090618827 - 0.9999589403j,
                -0.9892287385 + 0.1463779457j,
                0.5667187936 - 0.8239112871j,
                0.6205016526 - 0.7842051384j,
                -0.7477492359 + 0.6639812348j,
                -0.2221131667 - 0.9750208927j,
                -0.1545027834 - 0.9879923532j,
                -0.1006488398 - 0.9949220125j,
                -0.0553572413 - 0.9984666123j,
                -0.0144219837 - 0.9998959978j,
                0.0214944021 - 0.9997689687j,
                -0.3288697385 + 0.9443752936j,
                -0.9892570194 + 0.1461866945j,
                0.2802951114 + 0.9599138766j,
                -0.0124033172 - 0.9999230759j,
                -0.1584949482 + 0.9873597882j,
                -0.3362508336 + 0.9417724656j,
                -0.3040457710 - 0.9526574249j,
                -0.2101846230 - 0.9776617126j,
                -0.1303232768 - 0.9914715546j,
                -0.0573257962 - 0.9983555244j,
                0.0214944021 - 0.9997689687j,
                -0.1269297494 + 0.9919117091j,
            ]
        )
        actual = _parse_u_mat_file(block)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_almost_equal(np.abs(expected), np.ones_like(expected))
