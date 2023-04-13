from typing import Generic, Literal, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.potential.potential import Potential

_L0Cov = TypeVar("_L0Cov", covariant=True)
class PotentialBasisConfig(TypedDict, Generic[_L0Cov]):
    """Configures the generation of an explicit basis from a given potential."""

    potential:Potential[_L0Cov, Literal[1], Literal[1]]
    n: int

def get_sho_basis_config()
