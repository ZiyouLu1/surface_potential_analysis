from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.state_vector import (
    as_dual_vector,
    calculate_inner_product,
)
from surface_potential_analysis.wavepacket.conversion import (
    convert_wavepacket_to_fundamental_momentum_basis,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_all_eigenstates,
    get_bloch_state,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    get_wavepacket_sample_fractions,
)

from ._projection import _get_single_point_state, get_projection_coefficients

if TYPE_CHECKING:
    from surface_potential_analysis.axis.axis import TransformedPositionAxis
    from surface_potential_analysis.basis.basis import AxisWithLengthBasis
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.wavepacket import Wavepacket

    _B0Inv = TypeVar("_B0Inv", bound=AxisWithLengthBasis[Any])
    _PB0Inv = TypeVar(
        "_PB0Inv", bound=tuple[TransformedPositionAxis[Any, Any, Any], ...]
    )
    _B1Inv = TypeVar("_B1Inv", bound=AxisWithLengthBasis[Any])

    _S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


def _build_real_lattice_block(wavepacket: Wavepacket[_S0Inv, _B0Inv]) -> str:
    return f"""begin unit_cell_cart
{' '.join(str(x) for x in wavepacket["basis"][0].delta_x * 10**10)}
{' '.join(str(x) for x in wavepacket["basis"][1].delta_x * 10**10)}
{' '.join(str(x) for x in wavepacket["basis"][2].delta_x * 10**10)}
end unit_cell_cart"""


def _build_kpoints_block(wavepacket: Wavepacket[_S0Inv, _B0Inv]) -> str:
    fractions = get_wavepacket_sample_fractions(wavepacket["shape"])
    # TODO(matt): declare inline in python 3.12  # noqa: TD003, FIX002
    newline = "\n"
    return f"""mp_grid : {" ".join(str(x) for x in wavepacket['shape'])}
begin kpoints
{newline.join([f"{f0} {f1} {f2}" for (f0,f1,f2) in fractions.T])}
end kpoints"""


def build_win_file(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], *, postproc_setup: bool = True
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
    return f"""
{"postproc_setup = .true." if postproc_setup else ""}
auto_projections = .true.
write_u_matrices = .true.
num_wann = 1
{_build_real_lattice_block(wavepacket)}
{_build_kpoints_block(wavepacket)}
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
    wavepacket: Wavepacket[_S0Inv, _PB0Inv], k: tuple[int, int, int, int, int]
) -> str:
    k_0, k_1, g_0, g_1, g_2 = k
    m = calculate_inner_product(
        _get_offset_bloch_state(
            get_bloch_state(wavepacket, k_1 - 1),
            (g_0, g_1, g_2),
        ),
        as_dual_vector(get_bloch_state(wavepacket, k_0 - 1)),
    )
    return f"""{k_0} {k_1} {g_0} {g_1} {g_2}
{np.real(m)} {np.imag(m)}"""


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


def build_mmn_file(wavepacket: Wavepacket[_S0Inv, _B0Inv], nnkpts_file: str) -> str:
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
    num_bands = 1
    num_kpts = np.prod(wavepacket["shape"])
    (nntot, nnkpts) = _parse_nnkpts_file(nnkpts_file)
    converted = convert_wavepacket_to_fundamental_momentum_basis(wavepacket)
    newline = "\n"
    return f"""
{num_bands} {num_kpts} {nntot}
{newline.join(_build_mmn_file_block(converted, k) for k in nnkpts)}"""


def build_amn_file(
    wavepacket: Wavepacket[_S0Inv, _B0Inv],
    projection: StateVector[_B1Inv],
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
    num_bands = num_wann = 1
    num_kpts = np.prod(wavepacket["shape"])
    coefficients = np.conj(
        get_projection_coefficients(projection, get_all_eigenstates(wavepacket))
    )
    newline = "\n"
    return f"""
{num_bands} {num_kpts} {num_wann}
{newline.join([f"1 1 {i+1} {np.real(c)} {np.imag(c)}" for (i,c) in enumerate(coefficients)])}
"""


def _parse_u_mat_file(
    u_matrix_file: str,
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    variables = [line.strip().split("  ") for line in u_matrix_file.splitlines()[4::3]]
    return np.array([float(s[0]) + 1j * float(s[1]) for s in variables])  # type: ignore[no-any-return]


def localize_wavepacket_from_u_matrix_file(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], u_matrix_file: str
) -> Wavepacket[_S0Inv, _B0Inv]:
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
    return {
        "basis": wavepacket["basis"],
        "eigenvalues": wavepacket["eigenvalues"],
        "shape": wavepacket["shape"],
        "vectors": coefficients[:, np.newaxis] * wavepacket["vectors"],
    }


def _write_localization_files_wannier90(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], tmp_dir_path: Path, nnkp_file: str
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(build_win_file(wavepacket, postproc_setup=False))

    mmn_filename = tmp_dir_path / "spa.mmn"
    with mmn_filename.open("w") as f:
        f.write(build_mmn_file(wavepacket, nnkp_file))

    amn_filename = tmp_dir_path / "spa.amn"
    with amn_filename.open("w") as f:
        transformed = _get_single_point_state(wavepacket)
        f.write(build_amn_file(wavepacket, transformed))


def _write_setup_files_wannier90(
    wavepacket: Wavepacket[_S0Inv, _B0Inv], tmp_dir_path: Path
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(build_win_file(wavepacket, postproc_setup=True))


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
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Build Files for initial Setup
        _write_setup_files_wannier90(wavepacket, tmp_dir_path)
        input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Setup Files
        nnkp_filename = tmp_dir_path / "spa.nnkp"
        with nnkp_filename.open("r") as f:
            nnkp_file = f.read()

        # Build Files for localisation
        _write_localization_files_wannier90(wavepacket, tmp_dir_path, nnkp_file)
        input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Result files, and localize wavepacket
        u_mat_filename = tmp_dir_path / "spa_u.mat"
        with u_mat_filename.open("r") as f:
            u_mat_file = f.read()

    return localize_wavepacket_from_u_matrix_file(wavepacket, u_mat_file)
