import copper_100
import nickel_111

# def calculate_sho_wavefunction(config: EigenstateConfig, z_min, z_max, nz):
#     z_points = np.linspace(z_min, z_max, nz)
#     sho_potential = 0.5 * config["mass"] * config["sho_omega"] ** 2 * z_points**2
#     potential_hamiltonian = np.diag(sho_potential)
#     print(potential_hamiltonian)
#     # dk = 2 * np.pi / (z_max - z_min)
#     dz = (z_max - z_min) / (nz)
#     k_points = 2 * np.pi * np.fft.fftfreq(nz, dz)
#     kinetic_energy = (hbar * k_points) ** 2 / (2 * config["mass"])
#     kinetic_hamiltonian = np.diag(kinetic_energy)
#     print(kinetic_energy)
#     print(np.fft.ifft(kinetic_energy, axis=0))

#     hamiltonian = potential_hamiltonian
#     for iz1 in range(nz):
#         for iz2 in range(nz):
#             diz = iz1 - iz2
#             hamiltonian[iz1, iz2] = -(((2 * np.pi) / (diz * dz)) ** 2)
#     return hamiltonian


if __name__ == "__main__":
    # nickel_111.s5_overlap.calculate_overlap_factor()
    # copper_100.s5_overlap.calculate_overlap_factor()
    copper_100.s5_overlap_analysis.plot_overlap()
    # nickel_111.s5_overlap_analysis.plot_overlap()
