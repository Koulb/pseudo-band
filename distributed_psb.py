import numpy as np
from matplotlib import pyplot as plt

import sys

from multiprocessing import Pool
from time import time, sleep

from psb import PseudoBand

# Initial data

# number of states
states = 7

# lattice constant in bohr radii
lattice_constant = 10.683

# basis in units of 2 pi / a
basis = np.array([
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1]
])

ff_dict_sym = {
    3.0: -0.5 * 0.23,
    8.0: 0.5 * 0.01,
    11.0: 0.5 * 0.06
}

ff_dict_asym = {
    3.0: 0.5 * 0.07,
    4.0: 0.5 * 0.05,
    11.0: 0.5 * 0.01
}

bands_start = 0
bands_end = 8

print("GaAs band structure calculation")

# number of threads
num_threads = 5

# sample points per k-path corrected for proper parallelisation
k_sampling = 100
while((3.25 * k_sampling + 1) % num_threads != 0):
    k_sampling += 1
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

# time
start_time = time()

test_path = [lambd, delta, x_uk, sigma]
test_path_ed = np.array_split(np.vstack(test_path), num_threads)

# Calculation


def calculate_band(path):
    psd = PseudoBand(states, lattice_constant, ff_dict_sym, ff_dict_asym, basis)
    band = np.real(psd.band_structure(path, bands_start, bands_end))

    return band

print("Starting parallel calculation: ")
# Multi process
pool = Pool(num_threads)

results = []
for i, path in enumerate(test_path_ed):
    results.append([i, path, pool.apply_async(calculate_band, args=(path,))])

ncompleted = 0
ntotal = len(test_path_ed)
data_items = []
while ncompleted < ntotal:
    sleep(2.0)
    completed = []
    for itask in range(len(results)):
        if results[itask][2].ready():
            number = results[itask][0]
            path = results[itask][1]
            band = results[itask][2].get()
            data_items.append([number, band])
            sys.stdout.flush()
            completed.append(itask)
            ncompleted += 1
            print('thread # ', itask, ' is ready')
    completed.sort()
    completed.reverse()
    for itask in completed:
        results.pop(itask)

data_items.sort()
data_items = [item[1] for item in data_items]
bands = [np.concatenate(item) for item in np.stack(data_items, 1)]

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
plt.ylim([-4, 7])

# Experimental data
experiment_data = dict({'L3_': [0, -0.9517525485267431],
                        'G15o': [1, -0.016129032258064058],
                        'X5': [2, -2.2554112554112553],
                        'G15o_': [3.25, 0.045105432202205975],
                        'L1': [0, 1.6934785644463055],
                        'G1': [1, 1.435483870967742],
                        'X1': [2, 1.8091746962714694],
                        'G1_': [3.25, 1.432202206395754],
                        'L3': [0, 5.016059209607596],
                        'G15': [1, 4.516129032258064],
                        'X3': [2, 2.1156263091746963],
                        'G15_': [3.25, 4.561164641809802],
                        })

for key, value in experiment_data.items():
    plt.scatter(k_sampling * value[0], value[1], c='black')

plt.scatter(0, -10, c='black', label='Cohen and Bergstresser (1966)')
plt.title('Band structure of GaAs')
plt.legend()

print('Execution time is {} seconds'.format(round(time() - start_time), 1))

plt.show()
