from __future__ import annotations

from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)

from .s4_wavepacket import (
    MAXIMUM_POINTS,
    load_copper_wavepacket,
    load_normalized_copper_wavepacket_momentum,
)
from .surface_data import save_figure


def plot_wavepacket_points() -> None:
    wavepacket = load_copper_wavepacket(0)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def animate_copper_111_wavepacket() -> None:
    wavepacket = load_normalized_copper_wavepacket_momentum(0, (0, 0, 102), 0)
    fig, _, _anim0 = animate_wavepacket_x0x1(wavepacket)
    fig.show()

    wavepacket = load_normalized_copper_wavepacket_momentum(1, (8, 8, 103), 0)
    fig, _, _anim1 = animate_wavepacket_x0x1(wavepacket)
    fig.show()
    input()


def plot_wavepacket_at_z_origin() -> None:
    normalized = load_normalized_copper_wavepacket_momentum(0, (0, 0, 102), 0)

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

    normalized = load_normalized_copper_wavepacket_momentum(1, (8, 8, 103), 0)

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
        max_point = MAXIMUM_POINTS[band]
        normalized = load_normalized_copper_wavepacket_momentum(band, max_point, 0)

        fig, ax, _ = plot_wavepacket_x0x1(normalized, max_point[2], measure="abs")
        fig.show()
        ax.set_title("Plot of abs(wavefunction) for z=z max")
        save_figure(fig, f"wavepacket_grid_{band}.png")
    input()
