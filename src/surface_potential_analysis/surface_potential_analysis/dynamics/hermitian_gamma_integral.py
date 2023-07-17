from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload

import numpy as np
import scipy
from scipy.constants import electron_mass, elementary_charge, epsilon_0, hbar

if TYPE_CHECKING:
    from collections.abc import Callable

    from surface_potential_analysis._types import _FloatLike_co

_S0Inv = TypeVar("_S0Inv", bound=tuple[int, ...])


@overload
def _get_delta_e(
    k_0: np.ndarray[_S0Inv, np.dtype[np.float_]],
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]] | float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    ...


@overload
def _get_delta_e(
    k_0: float,
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]],
) -> np.ndarray[_S0Inv, np.dtype[np.float_]] | float:
    ...


@overload
def _get_delta_e(
    k_0: float,
    k_1: float,
) -> float:
    ...


def _get_delta_e(
    k_0: np.ndarray[_S0Inv, np.dtype[np.float_]] | float,
    k_1: np.ndarray[_S0Inv, np.dtype[np.float_]] | float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]] | float:
    e_0 = (hbar * k_0) ** 2 / (2 * electron_mass)
    e_1 = (hbar * k_1) ** 2 / (2 * electron_mass)
    return e_0 - e_1  # type: ignore[no-any-return]


def _get_fermi_occupation(
    k: np.ndarray[_S0Inv, np.dtype[np.float_]],
    *,
    k_f: float,
    boltzmann_energy: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    return 1 / (1 + np.exp((_get_delta_e(k, k_f)) / boltzmann_energy))  # type: ignore[no-any-return]


def _get_hopping_availability(
    k: np.ndarray[_S0Inv, np.dtype[np.float_]],
    *,
    omega: float,
    k_f: float,
    boltzmann_energy: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    k_3 = np.sqrt(k**2 - 2 * electron_mass * omega / (hbar**2))
    kwargs = {"k_f": k_f, "boltzmann_energy": boltzmann_energy}
    return _get_fermi_occupation(k, **kwargs) * (  # type: ignore[no-any-return]
        1 - _get_fermi_occupation(k_3, **kwargs)
    )


def get_hermitian_gamma_occupation_integrand(
    k: np.ndarray[_S0Inv, np.dtype[np.float_]],
    *,
    omega: float,
    k_f: float,
    boltzmann_energy: float,
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    """
    Get the integrand of the hermitian_gamma_occupation_integral.

    Parameters
    ----------
    k : np.ndarray[_S0Inv, np.dtype[np.float_]]
    omega : float
    k_f : float
    boltzmann_energy : float

    Returns
    -------
    np.ndarray[_S0Inv, np.dtype[np.float_]]
    """
    k_3 = np.sqrt(k**2 - 2 * electron_mass * omega / (hbar**2))
    kwargs = {"k_f": k_f, "boltzmann_energy": boltzmann_energy}
    return _get_fermi_occupation(k, **kwargs) * (  # type: ignore[no-any-return]
        1 - _get_fermi_occupation(k_3, **kwargs)
    )


def calculate_hermitian_gamma_occupation_integral(
    omega: _FloatLike_co, k_f: _FloatLike_co, boltzmann_energy: _FloatLike_co
) -> float:
    """
    Calculate int_k1 N1(1-N3) dk1.

    Parameters
    ----------
    omega : float
    k_f : float
    boltzmann_energy : float

    Returns
    -------
    float
    """
    d_k = 2 * boltzmann_energy * electron_mass / (hbar**2 * k_f)
    return scipy.integrate.quad(  # type: ignore[no-any-return]
        lambda k: get_hermitian_gamma_occupation_integrand(
            k, omega=omega, k_f=k_f, boltzmann_energy=boltzmann_energy
        ),
        k_f - 20 * d_k,
        k_f + 20 * d_k,
    )[0]


def _calculate_real_gamma_prefactor(k_f: float) -> float:
    return 4 * electron_mass * k_f**3 / (hbar**3 * (2 * np.pi) ** 3)  # type: ignore[no-any-return]


def _get_coulomb_potential(
    q: np.ndarray[_S0Inv, np.dtype[np.float_]]
) -> np.ndarray[_S0Inv, np.dtype[np.float_]]:
    bohr_radius = (
        4 * np.pi * epsilon_0 * hbar**2 / (elementary_charge**2 * electron_mass)
    )
    alpha = 2 / bohr_radius
    q_div_alpha = q / alpha
    small_q_limit = (
        -2
        * (elementary_charge**2 / (epsilon_0 * alpha**2))
        * (1 - 3 / 2 * (q_div_alpha) ** 2)
    )
    full_expression = (elementary_charge**2 / (epsilon_0 * q**2)) * (
        (1 / (1 + q_div_alpha**2)) ** 2 - 1
    )

    return np.where(np.isclose(q_div_alpha, 0), small_q_limit, full_expression)  # type: ignore[no-any-return]


def _get_hopping_potential_integrand(
    phi: np.ndarray[_S0Inv, np.dtype[np.float_]],
    *,
    k_f: float,
    overlap: Callable[
        [np.ndarray[_S0Inv, np.dtype[np.float_]]],
        np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
    ],
) -> np.ndarray[_S0Inv, np.dtype[np.complex_]]:
    q = k_f * np.sin(phi / 2)
    return _calculate_real_gamma_prefactor(k_f) * np.sin(phi) * _get_coulomb_potential(q) ** 2 * overlap(q).astype(np.complex_)  # type: ignore[no-any-return]


def _complex_quad(func: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    def real_func(x: Any) -> Any:  # noqa: ANN401
        return np.real(func(x))

    def imag_func(x: Any) -> Any:  # noqa: ANN401
        return np.imag(func(x))

    real_integral = scipy.integrate.quad(real_func, *args, **kwargs)
    imag_integral = scipy.integrate.quad(imag_func, *args, **kwargs)
    return (
        real_integral[0] + 1j * imag_integral[0],
        real_integral[1:],
        imag_integral[1:],
    )


def calculate_hermitian_gamma_potential_integral(
    k_f: float,
    overlap: Callable[
        [np.ndarray[_S0Inv, np.dtype[np.float_]]],
        np.ndarray[_S0Inv, np.dtype[np.complex_ | np.float_]],
    ],
) -> np.complex_:
    """
    Given the overlap as a function of |q|, calculate the potential integral in hermitian gamma.

    Parameters
    ----------
    k_f : float
    overlap : Callable[ [np.ndarray[_S0Inv, np.dtype[np.float_]]], np.ndarray[_S0Inv, np.dtype[np.float_]], ]

    Returns
    -------
    float
    """
    return _complex_quad(  # type: ignore[no-any-return]
        lambda k: _get_hopping_potential_integrand(k, k_f=k_f, overlap=overlap),
        0,
        np.pi,
    )[0]


def calculate_real_gamma_integral(
    *,
    omega: float,
    k_f: float,
    boltzmann_energy: float,
    overlap: Callable[
        [np.ndarray[_S0Inv, np.dtype[np.float_]]],
        np.ndarray[_S0Inv, np.dtype[np.complex_]],
    ],
) -> np.complex_:
    r"""Calculate the hermitian part of gamma.

    \gamma_{\vec{l}_1,\vec{l}_0}(\omega) =
    \frac{4 m_e k_f^3}{\hbar {(2\pi)}^3 }
    \int_{\phi=0}^\pi \sin{\phi}
    \tilde{V}^*_{\vec{l}_1}(|\vec{q}|) \tilde{V}_{\vec{l}_0}(|\vec{q}|)
    d\phi
    \int_{k} dk n(k)_{fd} (1 - n_{fd}(k'))
    """
    potential_integral = calculate_hermitian_gamma_potential_integral(k_f, overlap)
    occupation_integral = calculate_hermitian_gamma_occupation_integral(
        omega, k_f, boltzmann_energy
    )
    return potential_integral * occupation_integral
