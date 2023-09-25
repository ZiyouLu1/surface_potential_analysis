from __future__ import annotations

from typing import TYPE_CHECKING, Any

from surface_potential_analysis.potential.potential import (
    Potential,
    UnevenPotential3d,
    interpolate_uneven_potential,
    load_uneven_potential_json,
    normalize_potential,
    truncate_potential,
    undo_truncate_potential,
)

from .surface_data import get_data_path

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis3d,
    )
    from surface_potential_analysis.basis.stacked_basis import StackedBasisLike


def load_raw_copper_potential() -> UnevenPotential3d[Any, Any, Any]:
    path = get_data_path("copper_raw_energies.json")
    data = load_uneven_potential_json(path)
    vector = data["data"].reshape(*data["basis"].shape)
    vector = 0.5 * (vector + vector[::-1])
    vector = 0.5 * (vector + vector[::, ::-1])
    vector = 0.5 * (vector + vector.swapaxes(0, 1))

    data["data"] = vector.reshape(-1)
    return data


def load_relaxed_copper_potential() -> UnevenPotential3d[Any, Any, Any]:
    path = get_data_path("raw_energies_relaxed_sp.json")
    return load_uneven_potential_json(path)


def load_nc_raw_copper_potential() -> UnevenPotential3d[Any, Any, Any]:
    path = get_data_path("copper_nc_raw_energies.json")
    return load_uneven_potential_json(path)


def load_9h_copper_potential() -> UnevenPotential3d[Any, Any, Any]:
    path = get_data_path("copper_9h_raw_energies.json")
    return load_uneven_potential_json(path)


def load_clean_copper_data() -> UnevenPotential3d[Any, Any, Any]:
    data = load_raw_copper_potential()
    data = normalize_potential(data)
    return truncate_potential(data, cutoff=3e-18, n=6, offset=1e-20)


def get_interpolated_potential(
    shape: tuple[int, int, int]
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[Any],
        FundamentalPositionBasis3d[Any],
        FundamentalPositionBasis3d[Any],
    ]
]:
    data = load_raw_copper_potential()
    normalized = normalize_potential(data)

    # The Top site has such an insanely large energy
    # We must bring it down first
    truncated = truncate_potential(normalized, cutoff=1e-17, n=5, offset=1e-20)
    truncated = truncate_potential(truncated, cutoff=0.5e-18, n=1, offset=0)
    interpolated = interpolate_uneven_potential(truncated, shape)

    vector = interpolated["data"].reshape(*interpolated["basis"].shape)
    vector = 0.5 * (vector + vector[::-1])
    vector = 0.5 * (vector + vector[::, ::-1])
    vector = 0.5 * (vector + vector.swapaxes(0, 1))
    interpolated["data"] = vector.reshape(-1)

    return undo_truncate_potential(interpolated, cutoff=0.5e-18, n=1, offset=0)


def get_interpolated_potential_relaxed(
    shape: tuple[int, int, int]
) -> Potential[
    StackedBasisLike[
        FundamentalPositionBasis3d[Any],
        FundamentalPositionBasis3d[Any],
        FundamentalPositionBasis3d[Any],
    ]
]:
    data = load_relaxed_copper_potential()
    normalized = normalize_potential(data)

    truncated = truncate_potential(normalized, cutoff=0.1e-18, n=1, offset=0)
    interpolated = interpolate_uneven_potential(truncated, shape)
    return undo_truncate_potential(interpolated, cutoff=0.1e-18, n=1, offset=0)
