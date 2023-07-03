from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import electron_volt
from surface_potential_analysis.state_vector.eigenstate_collection_plot import (
    plot_energies_against_bloch_phase_1d,
)

from hydrogen_nickel_111.s3_eigenstates import get_eigenstate_collection_hydrogen


def generate_bandwidth_table() -> None:
    collection = get_eigenstate_collection_hydrogen((24, 24, 12))
    collection = get_eigenstate_collection_hydrogen((26, 26, 12))
    collection = get_eigenstate_collection_hydrogen((26, 26, 10))

    energies = np.min(collection["energies"], axis=0) - np.min(collection["energies"])
    energies_mev = energies * 1000 / (electron_volt)

    bandwidths = np.max(collection["energies"], axis=0) - np.min(
        collection["energies"], axis=0
    )
    bandwidths_mev = bandwidths * 1000 / (electron_volt)

    table = f"""
\\begin{{table*}}\\centering
    \\ra{{1.3}}
    \\begin{{tabular}}{{@{{}}rlrr@{{}}}}
        \\toprule
        Band & Type              & Energy   & Bandwidth \\\\
        \\midrule
        0    & FCC groundstate   & {energies_mev[0]:.1e} & {bandwidths_mev[0]:.1e}  \\\\
        1    & HCP groundstate   & {energies_mev[1]:.1e} & {bandwidths_mev[1]:.1e}  \\\\
        2    & FCC parallel      & {energies_mev[2]:.1e} & {bandwidths_mev[2]:.1e}  \\\\
        3    & FCC parallel      & {energies_mev[3]:.1e} & {bandwidths_mev[3]:.1e}  \\\\
        4    & HCP parallel      & {energies_mev[4]:.1e} & {bandwidths_mev[4]:.1e}  \\\\
        5    & HCP parallel      & {energies_mev[5]:.1e} & {bandwidths_mev[5]:.1e}  \\\\
        6    & FCC perpendicular & {energies_mev[6]:.1e} & {bandwidths_mev[6]:.1e}  \\\\
        7    & HCP perpendicular & {energies_mev[7]:.1e} & {bandwidths_mev[7]:.1e}  \\\\
        \\bottomrule
    \\end{{tabular}}
    \\caption{{
        Energy of the first eight bands of Hydrogen on the surface of Nickel,
        .. consistent with previous calculations
    }}
\\end{{table*}}"""
    print(table)  # noqa: T201


def plot_band_convergence_diagram() -> None:
    fig, ax = plt.subplots()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    collection = get_eigenstate_collection_hydrogen((24, 24, 12))
    for band in [0, 2, 3, 6]:
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label("FCC")
        ln.set_color(colors[0])

    for band in [1, 4, 5, 7]:
        _, _, ln = plot_energies_against_bloch_phase_1d(
            collection, np.array([1, 0, 0]), band=band, ax=ax
        )
        ln.set_label("HCP")
        ln.set_color(colors[1])

    (fcc_dummy,) = ax.plot([])
    fcc_dummy.set_color(colors[0])
    (hcp_dummy,) = ax.plot([])
    hcp_dummy.set_color(colors[1])
    ax.legend([fcc_dummy, hcp_dummy], ["FCC", "HCP"])
    fig.show()
    input()
