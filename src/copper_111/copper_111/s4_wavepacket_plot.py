from surface_potential_analysis.wavepacket.plot import (
    animate_wavepacket_x0x1,
    plot_wavepacket_sample_frequencies,
    plot_wavepacket_x0x1,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    load_wavepacket,
    normalize_wavepacket,
)

from .surface_data import get_data_path, save_figure


def plot_wavepacket_points() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)
    fig, _, _ = plot_wavepacket_sample_frequencies(wavepacket)

    fig.show()

    input()


def animate_copper_111_wavepacket() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)
    animate_wavepacket_x0x1(wavepacket)


def plot_wavepacket_at_z_origin() -> None:
    path = get_data_path("eigenstates_grid_0.json")
    wavepacket = load_wavepacket(path)
    normalized = normalize_wavepacket(wavepacket)

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_imag.png")

    path = get_data_path("eigenstates_grid_1.json")
    wavepacket = load_wavepacket(path)

    origin = (
        (util.delta_x0[0] + util.delta_x1[0]) / 3,
        (util.delta_x0[1] + util.delta_x1[1]) / 3,
        0,
    )
    normalized = normalize_wavepacket(wavepacket, origin)

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="abs")
    fig.show()
    ax.set_title("Plot of abs(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="real")
    fig.show()
    ax.set_title("Plot of real(wavefunction) for z=0")
    save_figure(fig, "wavepacket_grid_z_origin_real.png")

    fig, ax, _ = plot_wavepacket_x0x1(normalized, 0, measure="imag")
    fig.show()
    ax.set_title("Plot of imag(wavefunction) for z=0")

    input()
