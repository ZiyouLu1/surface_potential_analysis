from energy_data import (
    fill_surface_from_z_maximum,
    interpolate_energies_grid,
    load_raw_energy_data,
    normalize_energy,
    truncate_energy,
)
from plot_energy_data import (
    plot_x_direction_energy_data,
    plot_xz_plane_energy,
    plot_z_direction_energy_data,
)

if __name__ == "__main__":
    data = normalize_energy(load_raw_energy_data())
    plot_xz_plane_energy(data)
    data = fill_surface_from_z_maximum(data)
    # data = fill_subsurface_from_hollow_sample(data)
    # Spline 1D v=600, n=6
    truncated_data = truncate_energy(data, cutoff=3e-18, n=6, offset=1e-20)

    plot_z_direction_energy_data(data, truncated_data)
    # plot_x_direction_energy_data(data)
    # plot_x_direction_energy_data(truncated_data)
    # plot_xz_plane_energy(data)
    # plot_xz_plane_energy(truncated_data)
    plot_xz_plane_energy(data)
    interpolated = interpolate_energies_grid(truncated_data, shape=(20, 20, 100))
    plot_z_direction_energy_data(data, interpolated)
    plot_xz_plane_energy(interpolated)
    plot_x_direction_energy_data(data)
    plot_x_direction_energy_data(interpolated)

    # truncated2 = truncate_energy(data, cutoff=6e-18, n=6, offset=1e-20)
    # interpolated2 = interpolate_energies_grid(truncated2, shape=(21, 21, 50))
    # plot_xz_plane_energy(interpolated2)
    # plot_z_direction_energy_data(data, interpolated2)
    # plot_z_direction_energy_data(data, truncated2)
    # plot_x_direction_energy_data(interpolated2)
    input()
