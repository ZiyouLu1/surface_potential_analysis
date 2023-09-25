from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, cast

import numpy as np

from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.state_vector import (
    as_dual_vector,
    calculate_inner_product,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    StateVectorList,
    as_state_vector_list,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_bloch_state_vector,
    get_states_at_bloch_idx,
)
from surface_potential_analysis.wavepacket.localization.localization_operator import (
    get_localized_wavepackets,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    WavepacketList,
    get_wavepacket_sample_fractions,
    wavepacket_list_into_iter,
)

from ._projection import (
    get_single_point_state_for_wavepacket,
    get_state_projections_many_band,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.wavepacket.localization.localization_operator import (
        LocalizationOperator,
    )

    _SBL0 = TypeVar(
        "_SBL0",
        bound=StackedBasisLike[*tuple[Any, ...]],
    )
    _PB1Inv = TypeVar(
        "_PB1Inv",
        bound=FundamentalTransformedPositionBasis[Any, Any],
    )
    _FB0 = TypeVar("_FB0", bound=FundamentalBasis[Any])

    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[*tuple[Any, ...]])

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])


class ProjectionsBasis(TypedDict, Generic[_B0]):
    basis: StackedBasisLike[_B0]


@dataclass
class Wannier90Options(Generic[_B0]):
    projection: ProjectionsBasis[_B0] | StateVectorList[_B0, Any]
    num_iter: int = 10000
    conv_window: int = 3


def _build_real_lattice_block(
    delta_x: np.ndarray[tuple[int, int], np.dtype[np.float_]]
) -> str:
    # Wannier90 expects the wavefunction to be 3D, so we add the extra fake axis here
    delta_x_padded = np.eye(3)
    n_dim = delta_x.shape[0]
    delta_x_padded[:n_dim, :n_dim] = delta_x * 10**10

    newline = "\n"
    return f"""begin unit_cell_cart
{newline.join(' '.join(str(x) for x in o ) for o in delta_x_padded) }
end unit_cell_cart"""


# ! cSpell:disable
def _build_k_points_block(
    list_basis: StackedBasisLike[*tuple[BasisLike[Any, Any], ...]]
) -> str:
    n_dim = list_basis.ndim
    fractions = get_wavepacket_sample_fractions(list_basis)
    fractions_padded = np.zeros((3, fractions.shape[1]))
    fractions_padded[:n_dim] = fractions

    # mp_grid is 1 in the directions not included in the wavefunction
    mp_grid = np.ones(3, dtype=np.int_)
    mp_grid[:n_dim] = np.array(list_basis.shape)

    # TODO(matt): declare inline in python 3.12  # noqa: TD003, FIX002
    newline = "\n"
    return f"""mp_grid : {" ".join(str(x) for x in mp_grid)}
begin kpoints
{newline.join([ f"{f0} {f1} {f2}" for (f0,f1,f2) in fractions_padded.T])}
end kpoints"""


def _build_win_file(
    wavepackets: WavepacketList[_B0, _SB0, _SBL0],
    *,
    postproc_setup: bool = False,
    options: Wannier90Options[Any],
) -> str:
    """
    Build a postproc setup file.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    str
    """
    util = BasisUtil(wavepackets["basis"][1])
    has_projections = options.projection.get("data", None) is not None
    return f"""
{"postproc_setup = .true." if postproc_setup else ""}
num_iter = {options.num_iter}
conv_window = {options.conv_window}
write_u_matrices = .true.
{"auto_projections = .true."if has_projections  else "use_bloch_phases = .true."}
num_wann = {wavepackets["basis"][0][0].n}
{_build_real_lattice_block(util.delta_x_stacked)}
{_build_k_points_block(wavepackets["basis"][0][1])}
search_shells = 100
"""


# ! cSpell:enable


def _get_offset_bloch_state(
    state: StateVector[_SB0], offset: tuple[int, ...]
) -> StateVector[_SB0]:
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
    padded_shape = np.ones(3, dtype=np.int_)
    padded_shape[: state["basis"].ndim] = state["basis"].shape
    # Should be -offset if psi(k+b) = psi(k)
    vector = np.roll(
        (state["data"]).reshape(padded_shape),
        tuple(-o for o in offset),
        (0, 1, 2),
    ).reshape(-1)
    return {"basis": state["basis"], "data": vector}


def _build_mmn_file_block(
    wavepackets: WavepacketList[_B0, _SB0, StackedBasisLike[*tuple[_PB1Inv, ...]]],
    k: tuple[int, int, int, int, int],
) -> str:
    k_0, k_1, *offset = k
    block = f"{k_0} {k_1} {offset[0]} {offset[1]} {offset[2]}"
    for wavepacket_n in wavepacket_list_into_iter(wavepackets):
        for wavepacket_m in wavepacket_list_into_iter(wavepackets):
            mat = calculate_inner_product(
                _get_offset_bloch_state(
                    get_bloch_state_vector(wavepacket_n, k_1 - 1),
                    tuple(offset),
                ),
                as_dual_vector(get_bloch_state_vector(wavepacket_m, k_0 - 1)),
            )
            block += f"\n{np.real(mat)} {np.imag(mat)}"
    return block


def _parse_nnk_points_file(
    nnk_points_file: str,
) -> tuple[int, list[tuple[int, int, int, int, int]]]:
    block = re.search(
        "begin nnkpts((.|\n)+?)end nnkpts", nnk_points_file  # cSpell:disable-line
    )
    if block is None:
        msg = "Unable to find nnk_points block"
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
    wavepackets: WavepacketList[_B0, _SB0, StackedBasisLike[*tuple[_PB1Inv, ...]]],
    nnk_points_file: str,
) -> str:
    """
    Given a .nnkp file, generate the mmn file.  # cSpell:disable-line.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
    file : str

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
    """
    n_wavefunctions = wavepackets["basis"][0][0].n
    n_k_points = wavepackets["basis"][0][1].n
    (n_n_tot, nnk_points) = _parse_nnk_points_file(nnk_points_file)

    newline = "\n"
    return f"""
{n_wavefunctions} {n_k_points} {n_n_tot}
{newline.join(_build_mmn_file_block(wavepackets, k) for k in nnk_points)}"""


def _build_amn_file(
    wavepackets: WavepacketList[
        _B0,
        StackedBasisLike[*tuple[_FB0, ...]],
        StackedBasisLike[*tuple[_PB1Inv, ...]],
    ],
    projections: StateVectorList[_B1, _B2],
) -> str:
    """
    Build an amn file from a wavepacket.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]
    projection: StateVector[_B1Inv]

    Returns
    -------
    str
        for idx in range(wavepackets["basis"][0][1].n)
    """
    n_projections = projections["basis"][0].n
    n_wavefunctions = wavepackets["basis"][0][0].n
    n_k_points = wavepackets["basis"][0][1].n
    coefficients = np.array(
        [
            get_state_projections_many_band(
                get_states_at_bloch_idx(wavepackets, idx), projections
            )["data"]
            for idx in range(n_k_points)
        ]
    )
    stacked = coefficients.reshape(n_k_points, n_wavefunctions, n_projections)
    newline = "\n"
    newline.join(
        f"{m+1} {n+1} {k+1} {np.real(s)} {np.imag(s)}"
        for ((k, m, n), s) in np.ndenumerate(stacked)
    )
    return f"""
{n_wavefunctions} {n_k_points} {n_projections}
{newline.join(
    f"{m+1} {n+1} {k+1} {np.real(s)} {np.imag(s)}"
    for ((k, m, n), s) in np.ndenumerate(stacked)
)}
"""


def _parse_u_mat_file_block(
    block: list[str],
) -> np.ndarray[tuple[int], np.dtype[np.complex_]]:
    variables = [line.strip().split() for line in block]
    return np.array([float(s[0]) + 1j * float(s[1]) for s in variables])  # type: ignore[no-any-return]


def _get_localization_operator_from_u_mat_file(
    wavepackets_basis: StackedBasisLike[_B0, _SB0],
    projection_basis: _B1,
    u_matrix_file: str,
) -> LocalizationOperator[_SB0, _B1, _B0]:
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
    n_wavefunctions = int(lines[1].strip().split()[1])
    n_bands = int(lines[1].strip().split()[2])
    a = np.array(
        [
            [
                _parse_u_mat_file_block(
                    lines[
                        4 + n + n_wavefunctions * m :: 2 + (n_wavefunctions * n_bands)
                    ]
                )
                for n in range(n_wavefunctions)
            ]
            for m in range(n_bands)
        ]
    )
    return {
        "basis": StackedBasis(
            wavepackets_basis[1], StackedBasis(projection_basis, wavepackets_basis[0])
        ),
        "data": np.moveaxis(a, -1, 0).reshape(-1),
    }


def _write_setup_files_wannier90(
    wavepackets: WavepacketList[_B0, _SB0, _SBL0],
    tmp_dir_path: Path,
    *,
    options: Wannier90Options[_B1],
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(_build_win_file(wavepackets, postproc_setup=True, options=options))


def _write_localization_files_wannier90(
    tmp_dir_path: Path,
    n_nkp_file: str,
    wavepackets: WavepacketList[_B0, StackedBasisLike[*tuple[_FB0, ...]], _SBL0],
    *,
    options: Wannier90Options[_B1],
) -> None:
    win_filename = tmp_dir_path / "spa.win"
    with win_filename.open("w") as f:
        f.write(_build_win_file(wavepackets, options=options))

    converted = convert_state_vector_list_to_basis(
        wavepackets,
        stacked_basis_as_fundamental_momentum_basis(wavepackets["basis"][1]),
    )

    mmn_filename = tmp_dir_path / "spa.mmn"
    with mmn_filename.open("w") as f:
        f.write(_build_mmn_file(converted, n_nkp_file))

    if options.projection.get("data", None) is not None:
        projection = cast(StateVectorList[_B1, Any], options.projection)
        amn_filename = tmp_dir_path / "spa.amn"
        with amn_filename.open("w") as f:
            f.write(_build_amn_file(converted, projection))


def get_localization_operator_wannier90(
    wavepackets: WavepacketList[_B0, _SB0, _SBL0],
    *,
    options: Wannier90Options[_B1],
) -> LocalizationOperator[_SB0, _B1, _B0]:
    """
    Localizes a set of wavepackets using wannier 90, with a single point projection as an initial guess.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnk_pts and _u.mat file
    into the temporary directory created by the function.

    Also requires the wavepackets to have their samples in the fundamental basis

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SBL0]
        Wavepackets to localize
    projections : StateVectorList[_B1, _B2]
        Projections used in the localization procedure

    Returns
    -------
    WavepacketList[_B1, _SB0, _SBL0]
        Localized wavepackets, with each wavepacket correspondign to a different projection
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        # Build Files for initial Setup
        _write_setup_files_wannier90(wavepackets, tmp_dir_path, options=options)
        input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Setup Files
        n_nkp_filename = tmp_dir_path / "spa.nnkp"  # cSpell:disable-line
        with n_nkp_filename.open("r") as f:
            n_nkp_file = f.read()

        # Build Files for localisation
        _write_localization_files_wannier90(
            wavepackets, tmp_dir_path, n_nkp_file, options=options  # type: ignore should be fundamental basis, but we have no way of ensuring this in the type system
        )
        input(f"Run Wannier 90 in {tmp_dir_path}")

        # Load Result files, and localize wavepacket
        u_mat_filename = tmp_dir_path / "spa_u.mat"
        with u_mat_filename.open("r") as f:
            u_mat_file = f.read()

    return _get_localization_operator_from_u_mat_file(
        wavepackets["basis"][0], options.projection["basis"][0], u_mat_file
    )


def localize_wavepacket_wannier90(
    wavepackets: WavepacketList[_B0, _SB0, _SBL0],
    *,
    options: Wannier90Options[_B1],
) -> WavepacketList[_B1, _SB0, _SBL0]:
    """
    Localizes a set of wavepackets using wannier 90, with a single point projection as an initial guess.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnk_pts and _u.mat file
    into the temporary directory created by the function.

    Also requires the wavepackets to have their samples in the fundamental basis

    Parameters
    ----------
    wavepackets : WavepacketList[_B0, _SB0, _SBL0]
        Wavepackets to localize
    projections : StateVectorList[_B1, _B2]
        Projections used in the localization procedure

    Returns
    -------
    WavepacketList[_B1, _SB0, _SBL0]
        Localized wavepackets, with each wavepacket correspondign to a different projection
    """
    operator = get_localization_operator_wannier90(wavepackets, options=options)
    return get_localized_wavepackets(wavepackets, operator)


def localize_wavepacket_wannier90_sp_projections(
    wavepackets: WavepacketList[_B0, _SB0, _SBL0],
) -> WavepacketList[FundamentalBasis[int], _SB0, _SBL0]:
    """
    Localizes a wavepacket using wannier 90, with a single point projection as an initial guess.

    Note this requires a user to manually run wannier90 and input the resulting wannier90 nnk_pts and _u.mat file
    into the temporary directory created by the function.

    Parameters
    ----------
    wavepacket : Wavepacket[_B0Inv, _B0Inv]

    Returns
    -------
    Wavepacket[_B0Inv, _B0Inv]
        The localized wavepacket
    """
    projection = as_state_vector_list(
        get_single_point_state_for_wavepacket(wavepacket)
        for wavepacket in wavepacket_list_into_iter(wavepackets)
    )
    return localize_wavepacket_wannier90(
        wavepackets, options=Wannier90Options(projection)
    )
