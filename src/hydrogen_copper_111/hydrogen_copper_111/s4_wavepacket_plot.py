from __future__ import annotations

from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)

from .s4_wavepacket import (
    get_two_point_normalized_wavepacket_hydrogen,
    get_wavepacket_hydrogen,
)
from .surface_data import save_figure


def plot_wavepacket_points() -> None:
    wavepacket = get_wavepacket_hydrogen(0)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def animate_copper_111_wavepacket() -> None:
    wavepacket = get_two_point_normalized_wavepacket_hydrogen(0)
    fig, _, _anim0 = animate_wavepacket_x0x1(wavepacket)
    fig.show()

    wavepacket = get_two_point_normalized_wavepacket_hydrogen(1)
    fig, _, _anim1 = animate_wavepacket_x0x1(wavepacket)
    fig.show()
    input()


def plot_wavepacket_at_z_origin() -> None:
    normalized = get_two_point_normalized_wavepacket_hydrogen(0)

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 102, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 102, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 102, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_imag.png")

    normalized = get_two_point_normalized_wavepacket_hydrogen(1)

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 103, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 103, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 103, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")

    input()


def plot_wavepacket_at_maximum_points() -> None:
    for band in range(20):
        normalized = get_two_point_normalized_wavepacket_hydrogen(band)

        fig, ax, _ = plot_wavepacket_x0x1(normalized, 103, measure="abs")
        fig.show()
        ax.set_title("Plot of abs(wavefunction) for z=z max")
        save_figure(fig, f"wavepacket_grid_{band}.png")
    input()
