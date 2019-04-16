import random
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


class Haldane:
    def __init__(self):
        self.t = 1
        self.lamb = 0
        self.V = 0.3
        self.sign = 1  # FOR BERRY PHASE : IF = 0 then negative vector, else: positive
        self.a1 = 0.5 * np.array([-3 ** 0.5, 3])  # translation vector1 in position domain
        self.a2 = 0.5 * np.array([3 ** 0.5, 3])  # translation vector2 in position domain

    """hamiltonian for a given k"""

    def define_hamiltonian(self, k1, k2):
        ab = self.t * (1 + np.exp(-1j * k1) + np.exp(-1j * k2))
        ba = np.conj(ab)
        aa = self.V + 1j * self.lamb * (-np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
                                        + np.exp(-1j * (k1 - k2)) - np.exp(-1j * k1) + np.exp(-1j * k2))
        bb = -self.V + self.lamb * (np.exp(1j * (k1 - k2)) + np.exp(1j * k1) - np.exp(1j * k2)
                                    - np.exp(1j * (k1 - k2)) + np.exp(-1j * k1) - np.exp(-1j * k2))

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

        ax.scatter3D(x, y, z1, c=z1, cmap='plasma')
        ax.scatter3D(x, y, z2, c=z2, cmap='plasma')

        plt.title("Energy bands in 3D, $\lambda$ = 0, V = 0.3", fontsize=label_font_size)
        ax.set_xlabel("$k_x$", fontsize=label_font_size)
        ax.set_ylabel("$k_y$", fontsize=label_font_size)
        ax.set_zlabel("E", fontsize=label_font_size)
        plt.show()

    def draw_energy_plot_path(self):
        label_font_size = 20
        x, y1, y2 = self.get_eigenvalues_for_plot_path()

        plt.plot(x, y1, 'o', markersize=3, color='red')
        plt.plot(x, y2, 'o', markersize=3, color='red')

        plt.title("Energy path in brillouin zone, $\lambda$ = 0, V = 0.3", fontsize=label_font_size)
        plt.xticks([0, 2.42, 3.03, 4.86], ('$\Gamma$', 'K', 'M', '$\Gamma$'), fontsize=label_font_size)
        plt.ylabel("E", fontsize=label_font_size)
        plt.show()

    def get_berry_phase(self, vec_k1, vec_k2, vec_k3, vec_k4):
        k1x = np.vdot(vec_k1, self.a1)
        k1y = np.vdot(vec_k1, self.a2)

        k2x = np.vdot(vec_k2, self.a1)
        k2y = np.vdot(vec_k2, self.a2)

        k3x = np.vdot(vec_k3, self.a1)
        k3y = np.vdot(vec_k3, self.a2)

        k4x = np.vdot(vec_k4, self.a1)
        k4y = np.vdot(vec_k4, self.a2)

        u1 = np.linalg.eigh(self.define_hamiltonian(k1x, k1y))[1]
        u2 = np.linalg.eigh(self.define_hamiltonian(k2x, k2y))[1]
        u3 = np.linalg.eigh(self.define_hamiltonian(k3x, k3y))[1]
        u4 = np.linalg.eigh(self.define_hamiltonian(k4x, k4y))[1]

        dot_prod12 = np.vdot(u1[:, self.sign], u2[:, self.sign])
        dot_prod23 = np.vdot(u2[:, self.sign], u3[:, self.sign])
        dot_prod34 = np.vdot(u3[:, self.sign], u4[:, self.sign])
        dot_prod41 = np.vdot(u4[:, self.sign], u1[:, self.sign])

        berry_phase = np.angle(dot_prod12 * dot_prod23 * dot_prod34 * dot_prod41
                               / np.abs(dot_prod12 * dot_prod23 * dot_prod34 * dot_prod41))

        berry_phase /= np.linalg.norm(vec_k1 - vec_k2) * np.linalg.norm(vec_k3 - vec_k4)
        return berry_phase

    def get_berry_curvature(self):
        kx_array, ky_array, b_curvature  = list(), list(), list()

        k_step = 0.05
        area_of_loop = k_step ** 2

        for kx in np.arange(-np.pi, np.pi, k_step):
            for ky in np.arange(-np.pi, np.pi, k_step):
                vec_k1 = np.array([kx, ky])
                vec_k2 = np.array([kx - k_step, ky])
                vec_k3 = np.array([kx - k_step, ky - k_step])
                vec_k4 = np.array([kx, ky - k_step])

                b_curvature.append(self.get_berry_phase(vec_k1, vec_k2, vec_k3, vec_k4))

                kx_array.append(kx)
                ky_array.append(ky)

        return kx_array, ky_array, b_curvature

    def plot_berry_curvature3d(self):
        label_font_size = 20
        x, y, z = self.get_berry_curvature()

        ax = plt.axes(projection='3d')

        ax.scatter3D(x, y, z, c=z, cmap='rainbow')
        # ax.plot_trisurf(x, y, z)

        plt.title("Berry curvature", fontsize=label_font_size)
        ax.set_xlabel("$k_x$", fontsize=label_font_size)
        ax.set_ylabel("$k_y$", fontsize=label_font_size)
        ax.set_zlabel("$F_{12}^n$", fontsize=label_font_size)

        plt.figure(2)
        temp = np.tile(z, (len(z), 1))
        im = plt.imshow(np.array(np.transpose(temp)), cmap='rainbow', extent=(-np.pi, np.pi, np.pi, -np.pi),
                        interpolation='bilinear')
        plt.colorbar(im)
        plt.show()

if __name__ == '__main__':
    test = Haldane()
    # test.draw_energy_plot3d()
    # test.draw_energy_plot_path()
    test.plot_berry_curvature3d()

