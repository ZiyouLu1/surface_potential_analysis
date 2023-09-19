from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def get_out_path(filename: str) -> Path:
    out_folder_env = os.getenv("OUT_FOLDER")
    out_folder = (
        Path(out_folder_env)
        if out_folder_env is not None
        else Path(__file__).parent.parent.parent.parent.parent / "out"
    )
    return out_folder / "hydrogen_platinum_111" / filename


def save_figure(fig: Figure, filename: str) -> None:
    path = get_out_path(filename)
    fig.savefig(path)  # type: ignore doesn't support PathLike


def get_data_path(filename: str) -> Path:
    data_folder_env = os.getenv("DATA_FOLDER")
    data_folder = (
        Path(data_folder_env)
        if data_folder_env is not None
        else Path(__file__).parent.parent.parent.parent.parent / "data"
    )
    return data_folder / "hydrogen_platinum_111" / filename
