from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.util import AxisWithLengthBasisUtil, BasisUtil
from surface_potential_analysis.state_vector.state_vector import (
    as_dual_vector,
    calculate_inner_product,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_fundamental_momentum_basis,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_all_states,
    get_bloch_state_vector,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_sample_fractions,
)

from ._projection import _get_single_point_state, get_projection_coefficients

if TYPE_CHECKING:
    from collections.abc import Sequence

    from surface_potential_analysis.axis.axis import TransformedPositionAxis
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.wavepacket import (
        Wavepacket,
    )

    _B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
    _PB0Inv = TypeVar(
        "_PB0Inv", bound=tuple[TransformedPositionAxis[Any, Any, Any], ...]
    )
    _B1Inv = TypeVar("_B1Inv", bound=AxisWithLengthBasis[Any])

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def _build_real_lattice_block(
    delta_x: np.ndarray[tuple[int, int], np.dtype[np.float_]]
) -> str:
    newline = "\n"
    return f"""begin unit_cell_cart
{newline.join(' '.join(str(x) for x in o ) for o in (delta_x* 10**10)) }
end unit_cell_cart"""


def _build_kpoints_block(shape: _S0Inv) -> str:
    fractions = get_wavepacket_sample_fractions(shape)
    # TODO(matt): declare inline in python 3.12  # noqa: TD003, FIX002
    newline = "\n"
    return f"""mp_grid : {" ".join(str(x) for x in shape)}
begin kpoints
{newline.join([f"{f0} {f1} {f2}" for (f0,f1,f2) in fractions.T])}
end kpoints"""


def _build_win_file(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]], *, postproc_setup: bool = False
) -> str:
    """
    Build a postproc setup file.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    str
    """
    util = AxisWithLengthBasisUtil(wavepackets[0]["basis"])
    return f"""
{"postproc_setup = .true." if postproc_setup else ""}
auto_projections = .true.
write_u_matrices = .true.
num_wann = {len(wavepackets)}
{_build_real_lattice_block(util.delta_x)}
{_build_kpoints_block(wavepackets[0]["shape"])}
search_shells = 48
"""


def _get_offset_bloch_state(
    state: StateVector[_B0Inv], offset: tuple[int, ...]
) -> StateVector[_B0Inv]:
    """
    Get the bloch state corresponding to the bloch k offset by 'offset'.

    Since the wavefunction is identical, this is just the rolled state vector

    Parameters
    ----------
    state : StateVector[_B0Inv]
        _description_
    offset : tuple[int, ...]
        _description_

    Returns
    -------
    StateVector[_B0Inv]
        _description_
    """
    util = BasisUtil(state["basis"])
    # Should be -offset if psi(k+b) = psi(k)
    vector = np.roll(
        (state["vector"]).reshape(util.shape),
        tuple(-o for o in offset),
        tuple(range(util.ndim)),
    ).reshape(-1)
    return {"basis": state["basis"], "vector": vector}


def _build_mmn_file_block(
    wavepackets: Sequence[Wavepacket[_S0Inv, _PB0Inv]],
    k: tuple[int, int, int, int, int],
) -> str:
    k_0, k_1, *offset = k
    block = f"{k_0} {k_1} {offset[0]} {offset[1]} {offset[2]}"
    for wavepacket_n in wavepackets:
        for wavepacket_m in wavepackets:
            mat = calculate_inner_product(
                _get_offset_bloch_state(
                    get_bloch_state_vector(wavepacket_n, k_1 - 1),
                    tuple(offset),
                ),
                as_dual_vector(get_bloch_state_vector(wavepacket_m, k_0 - 1)),
            )
            block += f"\n{np.real(mat)} {np.imag(mat)}"
    return block


def _parse_nnkpts_file(
    nnkpts_file: str,
) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    block = re.search("begin nnkpts((.|\n)+?)end nnkpts", nnkpts_file)
    if block is None:
        msg = "Unable to find nnkpoints block"
        raise Exception(msg)  # noqa: TRY002
    lines = block.group(1).strip().split("\n")
    first_element = int(lines[0])
    data: list[tuple[int, int, int, int, int]] = []

    for line in lines[1:]:
        values = line.split()
        data_entry = tuple(map(int, values[:5]))
        data.append(data_entry)  # type: ignore[arg-type]

    return first_element, data


def _build_mmn_file(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]], nnkpts_file: str
) -> str:
    """
    Given a .nnkp file, generate the mmn file.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    file : str

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
    """
    num_kpts = np.prod(wavepackets[0]["shape"])
    (nntot, nnkpts) = _parse_nnkpts_file(nnkpts_file)
    converted = [
        convert_wavepacket_to_fundamental_momentum_basis(wavepacket)
        for wavepacket in wavepackets
    ]
    newline = "\n"
    return f"""
{len(wavepackets)} {num_kpts} {nntot}
{newline.join(_build_mmn_file_block(converted, k) for k in nnkpts)}"""


def _build_amn_file(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]],
    projections: Sequence[StateVector[_B1Inv]],
) -> str:
    """
    Build an amn file from a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    projection: StateVector[_B1Inv]

    Returns
    -------
    str
    """
    num_kpts = np.prod(wavepackets[0]["shape"])
    eigenstates = [get_all_states(wavepacket) for wavepacket in wavepackets]
    # Note: np.conj as get_projection_coefficients is the wrong way round
    coefficients = [
        (m, n, np.conj(get_projection_coefficients(projection, states)))
        for (m, states) in enumerate(eigenstates)
        for (n, projection) in enumerate(projections)
    ]
    newline = "\n"
    return f"""
{len(wavepackets)} {num_kpts} {len(projections)}
{newline.join(newline.join(f"{m+1} {n+1} {k+1} {np.real(v)} {np.imag(v)}" for (k,v) in enumerate(c)) for (m,n,c) in coefficients)}
"""


def _parse_u_mat_file_block(
    block: list[str],
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    variables = [line.strip().split() for line in block]
    return np.array([float(s[0]) + 1j * float(s[1]) for s in variables])  # type: ignore[no-any-return]


def _parse_u_mat_file(
    u_matrix_file: str,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]:
    """
    Get the u_mat coefficients, indexed such that U_{m,n}^k === out[n, m, k].

    Parameters
    ----------
    u_matrix_file : str

    Returns
    -------
    np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]
    """
    lines = u_matrix_file.splitlines()
    num_wann = int(lines[1].strip().split()[1])
    return np.array(  # type: ignore[no-any-return]
        [
            [
                _parse_u_mat_file_block(
                    lines[4 + n + num_wann * m :: 2 + (num_wann * num_wann)]
                )
                for n in range(num_wann)
            ]
            for m in range(num_wann)
        ]
    )


def localize_wavepacket_from_u_matrix_file(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]], u_matrix_file: str
) -> list[Wavepacket[_S0Inv, _B0Inv]]:
    """
    Given a _u.mat file, localize the wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]
    file : str

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
    """
    coefficients = _parse_u_mat_file(u_matrix_file)
    vectors = np.array([w["vectors"] for w in wavepackets])
    return [
        {
            "basis": wavepackets[0]["basis"],
            "shape": wavepackets[0]["shape"],
            "vectors": np.sum(coefficient[:, :, np.newaxis] * vectors, axis=0),
        }
        for coefficient in coefficients
    ]


def _write_setup_files_wannier90(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]], tmp_dir_path: Path
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(_build_win_file(wavepackets, postproc_setup=True))


def _write_localization_files_wannier90(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]],
    tmp_dir_path: Path,
    nnkp_file: str,
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(_build_win_file(wavepackets))

    mmn_filename = tmp_dir_path / "spa.mmn"
    with mmn_filename.open("w") as f:
        f.write(_build_mmn_file(wavepackets, nnkp_file))

    amn_filename = tmp_dir_path / "spa.amn"
    with amn_filename.open("w") as f:
        projections = [
            _get_single_point_state(wavepacket) for wavepacket in wavepackets
        ]
        f.write(_build_amn_file(wavepackets, projections))


def localize_wavepacket_wannier90_many_band(
    wavepackets: Sequence[Wavepacket[_S0Inv, _B0Inv]]
) -> list[Wavepacket[_S0Inv, _B0Inv]]:
    """
    Localizes a set of wavepackets using wannier 90, with a single point projection as an initial guess.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnkpts and _umat file
    into the temporary directory created by the function.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
        The localized wavepacket
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Build Files for initial Setup
        _write_setup_files_wannier90(wavepackets, tmp_dir_path)
        input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Setup Files
        nnkp_filename = tmp_dir_path / "spa.nnkp"
        with nnkp_filename.open("r") as f:
            nnkp_file = f.read()

        # Build Files for localisation
        _write_localization_files_wannier90(wavepackets, tmp_dir_path, nnkp_file)
        input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Result files, and localize wavepacket
        u_mat_filename = tmp_dir_path / "spa_u.mat"
        with u_mat_filename.open("r") as f:
            u_mat_file = f.read()

    return localize_wavepacket_from_u_matrix_file(wavepackets, u_mat_file)


def localize_wavepacket_wannier90(
    wavepacket: Wavepacket[_S0Inv, _B0Inv]
) -> Wavepacket[_S0Inv, _B0Inv]:
    """
    Localizes a wavepacket using wannier 90, with a single point projection as an initial guess.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnkpts and _umat file
    into the temporary directory created by the function.

    Parameters
    ----------
    wavepacket : Wavepacket[_S0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_S0Inv, _B0Inv]
        The localized wavepacket
    """
    return localize_wavepacket_wannier90_many_band([wavepacket])[0]
