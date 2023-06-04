from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.eigenstate.eigenstate_collection import (
    EigenstateColllection,
    calculate_eigenstate_collection,
)
from surface_potential_analysis.util.decorators import npy_cached

from .s2_hamiltonian import get_hamiltonian
from .surface_data import get_data_path

if TYPE_CHECKING:
    from pathlib import Path

    from surface_potential_analysis.axis.axis import FundamentalMomentumAxis1d
    from surface_potential_analysis.hamiltonian.hamiltonian import Hamiltonian

_L0Inv = TypeVar("_L0Inv", bound=int)


def _get_eigenstate_collection_cache(shape: tuple[_L0Inv]) -> Path:
    return get_data_path(f"eigenstates/eigenstates_{shape[0]}.npy")


@npy_cached(_get_eigenstate_collection_cache, allow_pickle=True)
def get_eigenstate_collection(
    shape: tuple[_L0Inv],
) -> EigenstateColllection[tuple[FundamentalMomentumAxis1d[_L0Inv]], Literal[5]]:
    h = get_hamiltonian(shape)
    util = BasisUtil(h["basis"])
    bloch_phases = np.linspace(0, (util.dk[0]) / 2, 5).reshape(5, 1)

    def hamiltonian_generator(
        bloch_phase: np.ndarray[tuple[Literal[1]], np.dtype[np.float_]]
    ) -> Hamiltonian[tuple[FundamentalMomentumAxis1d[_L0Inv]]]:
        return get_hamiltonian(shape=shape, bloch_phase=bloch_phase)

    return calculate_eigenstate_collection(
        hamiltonian_generator, bloch_phases, subset_by_index=(0, 10)  # type: ignore[arg-type]
    )
