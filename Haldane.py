import random
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0.05
        self.V = 0
        self.a1 = 0.5 * np.array([-3 ** 0.5, 3])  # translation vector1 in position domain
        self.a2 = 0.5 * np.array([3 ** 0.5, 3])  # translation vector2 in position domain

    """hamiltonian for a given k"""

    def define_hamiltonian(self, k1, k2):
        ab = self.t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
        ba = np.conj(ab)
        aa = self.V + 1j * self.lamb * (-np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
                                        + np.exp(-1j * (k1 - k2)) - np.exp(-1j * k1) + np.exp(-1j * k2))
        bb = -aa

        hamiltonian = np.array([[aa, ab],
                                [ba, bb]])
        return hamiltonian

    def get_eigenvalues_for_plot3d(self):
        eigenvalues_array_0, eigenvalues_array_1, kx_array, ky_array = list(), list(), list(), list()
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
        eigenvalues_array_0, eigenvalues_array_1, k_array = list(), list(), list()
        k_step = 0.01
        k_counter = 0
        ky = 0

        for kx in np.arange(0, 4 * 3**0.5 * np.pi / 9, k_step):
            k1 = np.dot(np.array([kx, ky]), self.a1)
            k2 = np.dot(np.array([kx, ky]), self.a2)

            eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
            eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
            k_counter += np.abs(k_step)
            k_array.append(k_counter)

        for kx in np.arange(4 * 3**0.5 * np.pi / 9, np.pi * 3**0.5 / 3, -k_step):
            ky = -3**0.5 * kx + 4*np.pi / 3
            k1 = np.dot(np.array([kx, ky]), self.a1)
            k2 = np.dot(np.array([kx, ky]), self.a2)

            eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
            eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
            k_counter += np.abs(k_step)
            k_array.append(k_counter)

        for kx in np.arange(np.pi * 3**0.5 / 3, 0, -k_step):
            ky = 3**0.5 / 3 * kx
            k1 = np.dot(np.array([kx, ky]), self.a1)
            k2 = np.dot(np.array([kx, ky]), self.a2)

            eigenvalues_array_0.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][0])
            eigenvalues_array_1.append(np.linalg.eigh(self.define_hamiltonian(k1, k2))[0][1])
            k_counter += np.abs(k_step)
            k_array.append(k_counter)

        return k_array, eigenvalues_array_0, eigenvalues_array_1

    def draw_energy_plot3d(self):
        label_font_size = 20
        x, y, z1, z2 = self.get_eigenvalues_for_plot3d()

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

        plt.plot(x, y1, 'o', markersize=3, color='red')
        plt.plot(x, y2, 'o', markersize=3, color='red')

        plt.title("Energy path in brillouin zone", fontsize=label_font_size)
        plt.xticks([0, 2.42, 3.03, 4.86], ('$\Gamma$', 'K', 'M', '$\Gamma$'), fontsize=label_font_size)
        plt.ylabel("E", fontsize=label_font_size)
        plt.show()

    def compute_berry_curvature(self):
        k_array, eigenvectors_array_negative, eigenvectors_array_positive = get_eigenvectors(t, dt)

if __name__ == '__main__':
    test = Haldane()
    # test.draw_energy_plot3d()
    test.draw_energy_plot_path()

print(4 * 3**0.5 * np.pi / 9)