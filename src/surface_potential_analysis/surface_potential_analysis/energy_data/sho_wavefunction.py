import math

import numpy as np
import scipy.special
from numpy.typing import ArrayLike, NDArray
from scipy.constants import hbar


def calculate_sho_wavefunction(
    z_points: ArrayLike, sho_omega: float, mass: float, n: int
) -> NDArray:
    norm = (sho_omega * mass / hbar) ** 0.5
    normalized_z = np.array(z_points) * norm

    prefactor = math.sqrt((norm / (2**n)) / (math.factorial(n) * math.sqrt(math.pi)))
    hermite = scipy.special.eval_hermite(n, normalized_z)
    exponential = np.exp(-np.square(normalized_z) / 2)
    return prefactor * hermite * exponential
