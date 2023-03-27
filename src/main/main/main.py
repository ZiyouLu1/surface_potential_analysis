import nickel_111

# def nkz_points(Nkz: int):
#     return np.array(np.rint(np.fft.fftfreq(Nkz, 1 / Nkz)), dtype=int)


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


# def get_ft_potential(potential):
#     fft_potential = np.fft.ifft(potential)

#     return fft_potential


# def _calculate_off_diagonal_entry(potential, ndkz) -> float:
#     """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
#     ft_pot_points = get_ft_potential(potential)[ndkz]

#     return ft_pot_points


# def _calculate_off_diagonal_energies(nkz, potential):
# kz_points = np.fft.fftshift(nkz_points(nkz))
# print(kz_points)
# hamiltonian = np.zeros(shape=(nkz, nkz), dtype=complex)
#
# for index1, nz0 in enumerate(kz_points):
# for index2, nz1 in enumerate(kz_points):
#             Number of jumps in units of dkx for this matrix element
# #
#             As k-> k+ Nx * dkx the ft potential is left unchanged
#             Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
#             Where Nx is the number of x points
# #
#             In reality we want to make sure ndkx < Nx (ie we should
#             make sure we generate enough points in the interpolation)
# ndkz = nz1 - nz0  # % len(potential)
# print(f"{index1}:{index2}", ndkz, len(potential), ndkz % len(potential))
#
# hamiltonian[index1, index2] = _calculate_off_diagonal_entry(potential, ndkz)
#
# return hamiltonian
#
#
# def test_calculate_off_diagonal_energies():
# potential = np.arange(4)
# e = _calculate_off_diagonal_energies(4, potential)
# print(4 * np.fft.ifft2(np.diag(potential), norm="ortho"))
# print(np.fft.fft2(np.diag(potential), norm="ortho"))
# a1 = np.fft.ifft(
# np.fft.fft(np.diag(potential), axis=0, norm="ortho"), axis=-1, norm="ortho"
# )
# a2 = np.fft.ifft(
# np.fft.fft(np.diag(potential), axis=-1, norm="ortho"), axis=0, norm="ortho"
# )
# a3 = np.fft.fft(
# np.fft.ifft(np.diag(potential), axis=0, norm="ortho"), axis=-1, norm="ortho"
# )
# a4 = np.fft.fft(
# np.fft.ifft(np.diag(potential), axis=-1, norm="ortho"), axis=0, norm="ortho"
# )
#
# np.testing.assert_array_equal(e, a1)
# np.testing.assert_array_equal(e, a4)
#
#     With interpolation
#     # potential = interpolation.interpolate_points_fftn(np.arange(5), (11,), (0,))
#     # print(np.fft.ifft(np.arange(5)), np.fft.ifft(potential))
#     e = _calculate_off_diagonal_energies(10, potential)
#     print(e)
# #
#     print(e * (16 * 3) / 63)
#     # e = _calculate_off_diagonal_energies(5, potential)
#     # print(e)
#     [[ 2.00000000e+00+0.00000000e+00j -5.00000000e-01-6.88190960e-01j  -5.00000000e-01-1.62459848e-01j -5.00000000e-01+1.62459848e-01j  -5.00000000e-01+6.88190960e-01j]
#      [-5.00000000e-01+6.88190960e-01j  2.00000000e+00+0.00000000e+00j  -5.00000000e-01-6.88190960e-01j  0                               -5.00000000e-01+1.62459848e-01j]
#      [-5.00000000e-01+1.62459848e-01j -5.00000000e-01+6.88190960e-01j   2.00000000e+00+0.00000000e+00j  0                               0                             ]
#      [-5.00000000e-01-1.62459848e-01j  0                                0                               2.00000000e+00+0.00000000e+00j  -5.00000000e-01-6.88190960e-01j]
#      [-5.00000000e-01-6.88190960e-01j -5.00000000e-01-1.62459848e-01j   0                              -5.00000000e-01+6.88190960e-01j   2.00000000e+00+0.00000000e+00j]]
# #
#     [[ 2.00000000e+00+0.00000000e+00j -5.00000000e-01-6.88190960e-01j  -5.00000000e-01-1.62459848e-01j  0                                0                             ]
#      [-5.00000000e-01+6.88190960e-01j  2.00000000e+00+0.00000000e+00j  -5.00000000e-01-6.88190960e-01j -5.00000000e-01-1.62459848e-01j   0                             ]
#      [-5.00000000e-01+1.62459848e-01j -5.00000000e-01+6.88190960e-01j   2.00000000e+00+0.00000000e+00j -5.00000000e-01-6.88190960e-01j  -5.00000000e-01-1.62459848e-01j]
#      [ 0                              -5.00000000e-01+1.62459848e-01j  -5.00000000e-01+6.88190960e-01j  2.00000000e+00+0.00000000e+00j  -5.00000000e-01-6.88190960e-01j]
#      [ 0                               0                               -5.00000000e-01+1.62459848e-01j -5.00000000e-01+6.88190960e-01j   2.00000000e+00+0.00000000e+00j]]
#     print(e * (16 * 3) / 63)
#     print(4 * np.fft.ifft2(np.diag(potential[::2])))
#     print(np.fft.fft2(np.diag(potential[::2])) / 4)
# a1 = np.fft.ifft(np.fft.fft(np.diag(potential[::2]), axis=0), axis=-1)
# a2 = np.fft.ifft(np.fft.fft(np.diag(potential[::2]), axis=-1), axis=0)
# a3 = np.fft.fft(np.fft.ifft(np.diag(potential[::2]), axis=0), axis=-1)
# a4 = np.fft.fft(np.fft.ifft(np.diag(potential[::2]), axis=-1), axis=0)
# print(potential[::2])
# print(a1)
# np.testing.assert_array_equal(e[:5:, :5:], a1, verbose=True)
# np.testing.assert_array_equal(e, a4)
#
# print("---------------")
# print(np.fft.ifft(np.arange(5)))
# print(np.fft.ifft(np.arange(10) / 2))
# print(interpolation.interpolate_real_points_along_axis_fourier(np.arange(5), 10))
# print(
# np.fft.ifft(
# interpolation.interpolate_real_points_along_axis_fourier(np.arange(5), 10)
# )
# )
# print(interpolation.interpolate_points_fftn(np.arange(5), (10,), (0,)))
# print(np.fft.ifft(interpolation.interpolate_points_fftn(np.arange(5), (10,), (0,))))
#
# fig, ax = plt.subplots()
# ax.plot(np.real(interpolation.interpolate_points_fftn(np.arange(5), (100,), (0,))))
# ax.plot(np.imag(interpolation.interpolate_points_fftn(np.arange(5), (100,), (0,))))
# fig.show()
# input()
#
#
# if __name__ == "__main__":
# copper_100.s3_eigenstates.generate_oversampled_eigenstates_data()
# copper_100.s3_eigenstates_plot.analyze_oversampling_effect()
# get_hamiltonian_from_potential(np.arange(1000).reshape(5, 20, 10))
#


def main() -> None:
    print("pass")
