import itertools
import functools
from time import time

import numpy as np
from matplotlib import pyplot as plt


class PseudoBand:
    def __init__(self, states, lat_constant, form_factors_sym, form_factors_asym, basis):
        self.states = states
        self.reciprocal_coeffs_ar = np.linspace(
            -states // 2 + 1, states // 2, states, dtype=int)
        self.reciprocal_coeffs = list(itertools.product(
            self.reciprocal_coeffs_ar, repeat=3))

        self.lat_constant = lat_constant
        self.form_factors_sym = form_factors_sym
        self.form_factors_asym = form_factors_asym
        self.basis = basis

        # Usefull constants
        self.kinetic_factor = (2 * np.pi / lat_constant)**2.0
        self.tau = 0.125 * np.ones(3)
        self.states3 = states**3
        self.states3_ar = np.linspace(
            0, self.states3 - 1, self.states3, dtype=int)

    @functools.lru_cache(343)
    def reciprocal_coords(self, number):

        middle = (self.states**3) // 2
        position = number + middle
        result = self.reciprocal_coeffs[position]
        h = result[0]
        k = result[1]
        l = result[2]

        return h, k, l

    def kinetic_energy(self, k_vector, G_vector):
        result = k_vector + G_vector

        return 0.5 * result @ result

    def potential_energy(self, G_vector, factor_sym, factor_asym):
        result = 0.0
        if(factor_sym):
            result += factor_sym * np.cos(2.0 * np.pi * G_vector @ self.tau)
        if (factor_asym):
            result += 1j * factor_asym * np.sin(2.0 * np.pi * G_vector @ self.tau)

        return result

    def hamiltonian(self, k_vector):

        h_matrix = np.empty((self.states3, self.states3), dtype=complex)
        middle_point = self.states3 // 2

        # self.states3_ar
        for row, col in itertools.product(range(self.states3), repeat=2):
            if row == col:
                G_vector = self.reciprocal_coords(
                    row - middle_point) @ self.basis
                h_matrix[row][col] = self.kinetic_factor * \
                    self.kinetic_energy(k_vector, G_vector)
            else:
                G_row_vector = self.reciprocal_coords(
                    row - middle_point) @ self.basis  # - middle_point
                G_col_vector = self.reciprocal_coords(
                    col - middle_point) @ self.basis  # - middle_point
                G_vector = G_row_vector - G_col_vector

                ff_sym = self.form_factors_sym.get(G_vector @ G_vector)
                ff_asym = self.form_factors_asym.get(G_vector @ G_vector)
                h_matrix[row][col] = self.potential_energy(
                    G_vector, ff_sym, ff_asym) if (ff_sym or ff_asym) else 0.0

        return h_matrix

    def band_structure(self, path, first_band=0, last_band=8):
        result = []

        for k_vector in np.vstack(path):
            h_matrix = self.hamiltonian(k_vector)
            bands = np.linalg.eigvals(h_matrix)
            bands.sort()
            result.append(bands[first_band:last_band])

        return np.stack(result, axis=-1)


if __name__ == "__main__":
    # # reciprocal_coords test
    # psd_band = PseudoBand(5, 1, 1, 1, 1)
    # print(psd_band.reciprocal_coords(62))

    # # kinetic_energy test
    # k_test = np.array([1, 1, 1])
    # G_test = np.array([1, 1, 1])
    # print(psd_band.kinetic_energy(k_test, G_test))

    # # potential energy test
    # factor = 1.0
    # print(psd_band.potential_energy(G_test, factor, factor))

    # Band_structure test
    states = 7

    # lattice constant in bohr radii
    lattice_constant = 10.26

    # symmetric form factors (From Rydbergs to Hartree)
    ff_dict_sym = {
        3.0: -0.5 * 0.21,
        8.0: 0.5 * 0.04,
        11.0: 0.5 * 0.08
    }

    ff_dict_asym = {}

    print("Si band structure calculation")

    # in units of 2 pi / a
    basis = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ])

    # sample points per k-path
    k_sampling = 100
    print("Total number of K-points = ", k_sampling)

    # symmetry points in the Brillouin zone
    G = np.array([0, 0, 0])
    L = np.array([1 / 2, 1 / 2, 1 / 2])
    K = np.array([3 / 4, 3 / 4, 0])
    X = np.array([0, 0, 1])
    W = np.array([1, 1 / 2, 0])
    U = np.array([1 / 4, 1 / 4, 1])

    # k-paths
    lambd = np.linspace(L, G, k_sampling, endpoint=False)
    delta = np.linspace(G, X, k_sampling, endpoint=False)
    x_uk = np.linspace(X, U, k_sampling // 4, endpoint=False)
    sigma = np.linspace(K, G, k_sampling + 1, endpoint=True)

    print("Starting calculation: ")
    # time
    start_time = time()

    psd_band = PseudoBand(states, lattice_constant, ff_dict_sym, ff_dict_asym, basis)
    test_path = [lambd, delta, x_uk, sigma]

    # Calculation
    bands = np.real(psd_band.band_structure(test_path, 0, 8))
    bands -= max(bands[3])
    to_ev = 27.2114
    plt.figure(figsize=(15, 9))
    colors = ['red', 'orange', 'green', 'blue', 'purple']

    xticks = k_sampling * np.array([0, 1, 2, 2.25, 3.25])
    for index, band in enumerate(bands):
        plt.plot(to_ev * np.array(band), '-', c=colors[index % len(colors)])

    plt.xticks(xticks, (r'$L$', r'$\Gamma$',
                        r'$X$', r'$K$', r'$\Gamma$'), fontsize=16)
    plt.xlabel('k points', fontsize=18)
    plt.yticks(fontsize=16)
    plt.ylabel('Energy, eV', fontsize=18)
    plt.ylim([-6, 6])

    # Experimental data 
    experiment_data = dict({'L3_': [0, -1.325056599279634],
                            'G25': [1, -0.028929928381481673],
                            'X4': [2, -3.0711649633039038],
                            'G25_': [3.25, -0.05985524267059894],
                            'L1': [0, 1.8041519229819052],
                            'G15': [1, 3.3530148063968905],
                            'X1': [2, 0.9237243149491787],
                            'G15_': [3.25, 3.3224062200296203],
                            'L3': [0, 3.987978274998385],
                            'G2_': [1, 3.832490203605378],
                            'G2': [3.25, 3.8016915804849996],
                            })

    for key, value in experiment_data.items():
        plt.scatter(k_sampling * value[0], value[1], c='black')

    plt.scatter(0, -10, c='black', label='Cohen and Bergstresser (1966)')
    plt.title('Band structure of Si')
    plt.legend()

    print('Execution time is {} seconds'.format(round(time() - start_time), 1))
    plt.show()
