import random
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0
        self.V = 0
        self.a1 = 0.5 * np.array([-3 ** 0.5, 3])  # translation vector1 in position domain
        self.a2 = 0.5 * np.array([3 ** 0.5, 3])  # translation vector2 in position domain

    """hamiltonian for a given k"""

    def define_hamiltonian(self, k1, k2):
        ab = self.t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
        ba = np.conj(ab)
        aa = self.V + 1j * self.lamb * (-np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
                                        + np.exp(-1j * (k1 - k2)) - np.exp(-1j * k1) + np.exp(-1j * k2))
        bb = np.conj(aa - self.V) - self.V

        hamiltonian = np.array([[aa, ab],
                                [ba, bb]])
        return hamiltonian

    def get_eigenvalues_for_plot(self):
        eigenvalues_array_0, eigenvalues_array_1, kx_array, ky_array = list(), list(), list(), list()
        ky = -np.pi
        k_step = 0.05
        for kx in np.arange(-np.pi, np.pi, k_step):  # defining grid loop
            for ky in np.arange(-np.pi, np.pi, k_step):
                k1 = np.dot(np.array([kx, ky]), self.a1)
                k2 = np.dot(np.array([kx, ky]), self.a2)
                eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
                eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
                kx_array.append(kx)
                ky_array.append(ky)

        return kx_array, ky_array, eigenvalues_array_0, eigenvalues_array_1

    def get_eigenvalues_for_plot_path(self):
        eigenvalues_array_0, eigenvalues_array_1, kx_array, ky_array = list(), list(), list(), list()
        ky = -np.pi
        k_step = 0.05
        for kx in np.arange(-np.pi, np.pi, k_step):  # defining grid loop
            for ky in np.arange(-np.pi, np.pi, k_step):
                k1 = np.dot(np.array([kx, ky]), self.a1)
                k2 = np.dot(np.array([kx, ky]), self.a2)
                eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
                eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
                kx_array.append(kx)
                ky_array.append(ky)

        return kx_array, ky_array, eigenvalues_array_0, eigenvalues_array_1

    def draw_energy_plot3d(self):
        label_font_size = 20
        x, y, z1, z2 = self.get_eigenvalues_for_plot()

        ax = plt.axes(projection='3d')

        ax.scatter3D(x, y, z1, c=z1, cmap='rainbow')
        ax.scatter3D(x, y, z2, c=z2, cmap='rainbow')

        plt.title("Energy bands in 3D", fontsize=label_font_size)
        ax.set_xlabel("$k_x$", fontsize=label_font_size)
        ax.set_ylabel("$k_y$", fontsize=label_font_size)
        ax.set_zlabel("E", fontsize=label_font_size)
        plt.show()

    def draw_energy_plot_path(self):
        label_font_size = 20
        x, y1, y2 = self.get_eigenvalues_for_plot_path()

        ax = plt.axes(projection='3d')

        plt.plot(x, y1, 'o', markersize=3, color='red')
        plt.plot(x, y2, 'o', markersize=3, color='red')

        plt.title("Energy path in brillouin zone", fontsize=label_font_size)
        ax.set_xlabel("$k_x$", fontsize=label_font_size)
        ax.set_ylabel("E", fontsize=label_font_size)
        plt.show()

if __name__ == '__main__':
    test = Haldane()
    test.draw_plot3d()
