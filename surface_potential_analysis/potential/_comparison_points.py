from __future__ import annotations


def get_111_comparison_points_x2(
    shape: tuple[int, int], offset: tuple[int, int] = (0, 0)
) -> dict[str, tuple[int, int]]:
    (nx0, nx1) = shape
    return {
        "Top Site": (((2 * nx0) // 3 + offset[0], (2 * nx1) // 3 + offset[1])),
        "Bridge Site": ((nx0 // 6 + offset[0], nx1 // 6 + offset[1])),
        "FCC Site": ((offset[0], offset[1])),
        "HCP Site": ((nx0 // 3 + offset[0], nx1 // 3 + offset[1])),
    }


def get_100_comparison_points_x2(
    shape: tuple[int, int], offset: tuple[int, int] = (0, 0)
) -> dict[str, tuple[int, int]]:
    (nx0, nx1) = shape
    return {
        "Hollow Site": (offset[0], offset[1]),
        "Bridge Site": (nx0 // 2 + offset[0], 0 + offset[1]),
        "Top Site": (nx0 // 2 + offset[0], nx1 // 2 + offset[1]),
    }
