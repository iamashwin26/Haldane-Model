import random

import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0.3
        self.V = 0
        self.a1 = 0.5 * np.array([-3 ** 0.5, 3])  # translation vector1 in position domain
        self.a2 = 0.5 * np.array([3 ** 0.5, 3])  # translation vector2 in position domain

    """hamiltonian for a given k"""

    def define_hamiltonian(self, k1, k2):
        ab = self.t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
        ba = np.conj(ab)
        aa = self.V + 1j * self.lamb * (-np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
                                        + np.exp(-1j * (k1 - k2)) - np.exp(-1j * k1) + np.exp(-1j * k2))
        bb = - aa

        hamiltonian = np.array([[aa, ab],
                                [ba, bb]])
        return hamiltonian

    def get_eigenvalues_for_plot(self):
        eigenvalues_array_0, eigenvalues_array_1, kx_array = list(), list(), list()
        ky = -np.pi
        k_step = 2

        for kx in np.arange(-np.pi, np.pi, k_step):  # defining grid loop
            k1 = np.dot(np.array([kx, ky]), self.a1)
            for ky in np.arange(-np.pi, np.pi, k_step):
                k2 = np.dot(np.array([kx, ky]), self.a2)

                eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
                eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
                kx_array.append(kx)

        return kx_array, eigenvalues_array_0, eigenvalues_array_1

    def draw_plot(self):
        x, y, z = self.get_eigenvalues_for_plot()
        # x, y = np.meshgrid(x, y)
        print('co to Z', z)
        print('co to Y', y)
        print('co to X', x)
        ax = plt.axes(projection='3d')

        ax.plot_trisurf(x, y, z)
        plt.show()


if __name__ == '__main__':
    test = Haldane()
    test.draw_plot()
