import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.basis import ExplicitBasis, PositionBasis
from surface_potential_analysis.basis.plot import plot_explicit_basis_states_x

from .s2_hamiltonian import generate_sho_basis


def _normalize_sho_basis(
    basis: ExplicitBasis[int, PositionBasis[int]]
) -> ExplicitBasis[int, PositionBasis[int]]:
    turning_point = basis["vectors"][
        np.arange(basis["vectors"].shape[0]),
        np.argmax(
            np.abs(basis["vectors"][:, : basis["vectors"].shape[1] // 2]), axis=1
        ),
    ]
    print(turning_point, np.angle(turning_point))

    normalized = np.exp(-1j * np.angle(turning_point))[:, np.newaxis] * basis["vectors"]
    return {"_type": "explicit", "parent": basis["parent"], "vectors": normalized}


def plot_sho_basis() -> None:
    """Plot the sho basis used for the nickel surface."""
    (infinate, finite) = generate_sho_basis(16)

    fig, ax = plt.subplots()

    normalized = _normalize_sho_basis(infinate)
    _, _, lines = plot_explicit_basis_states_x(normalized, ax=ax, measure="real")
    for line in lines:
        line.set_color("tab:blue")

    normalized = _normalize_sho_basis(finite)
    _, _, lines = plot_explicit_basis_states_x(normalized, ax=ax, measure="real")
    for line in lines:
        line.set_color("tab:orange")

    fig.show()
    input()
