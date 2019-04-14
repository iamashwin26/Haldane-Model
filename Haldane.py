
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


t = 1
lamb = 0.3
V = 0
a1 = 0.5 * np.array([-3 ** 0.5, 3])  # translation vector1 in position domain
a2 = 0.5 * np.array([3 ** 0.5, 3])  # translation vector2 in position domain

"""hamiltonian for a given k"""

def define_hamiltonian(k1, k2):
    global t
    global lamb
    global V
    ab = t * (1 + np.exp(-1j*k1) + np.exp(-1j*k2))
    ba = np.conj(ab)
    aa = V + 1j*lamb*(-np.exp(1j*(k1 - k2)) + np.exp(1j*k1) - np.exp(1j*k2)
                      + np.exp(-1j*(k1 - k2)) - np.exp(-1j*k1) + np.exp(-1j*k2))
    bb = - aa

    hamiltonian = np.array([[aa, ab],
                            [ba, bb]])
    return hamiltonian


def get_eigenvalues_for_plot_path():  # its just a scratch, path need to be build from zero
    global a1
    global a2
    eigenvalues_array_0 = []
    eigenvalues_array_1 = []
    kx_array = []
    ky_array = []
    ky = -np.pi
    k_step = 2

    for kx in np.arange(-np.pi, np.pi, k_step):  # defining grid loop
        k1 = np.dot(np.array([kx, ky]), a1)
        for ky in np.arange(-np.pi, np.pi, k_step):
            k2 = np.dot(np.array([kx, ky]), a2)

            eigenvalues_array_0.append(np.linalg.eigh(define_hamiltonian(k1, k2))[0][0])
            eigenvalues_array_1.append(np.linalg.eigh(define_hamiltonian(k1, k2))[0][1])

            ky_array.append(ky)
            kx_array.append(kx)
    return kx_array, ky_array, eigenvalues_array_0, eigenvalues_array_1, energy2d_array


def get_eigenvalues_for_plot():
    global a1
    global a2
    eigenvalues_array_0 = []
    eigenvalues_array_1 = []
    kx_array = []
    ky_array = []
    ky = -np.pi
    k_step = 0.1

    for kx in np.arange(-np.pi, np.pi, k_step):  # defining grid loop
        k1 = np.dot(np.array([kx, ky]), a1)
        for ky in np.arange(-np.pi, np.pi, k_step):
            k2 = np.dot(np.array([kx, ky]), a2)

            eigenvalues_array_0.append(np.linalg.eigh(define_hamiltonian(k1, k2))[0][0])
            eigenvalues_array_1.append(np.linalg.eigh(define_hamiltonian(k1, k2))[0][1])

            ky_array.append(ky)
            kx_array.append(kx)
    return kx_array, ky_array, eigenvalues_array_0, eigenvalues_array_1


def plot3d(x, y, z1, z2):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    X, Y = np.meshgrid(x, y)
    Z1 = np.tile(z1, (len(z1), 1))
    Z2 = np.tile(z2, (len(z2), 1))
    ax.plot_surface(X, Y, Z1, cmap='ocean')
    ax.plot_surface(X, Y, Z2, cmap='ocean')

    # ax.scatter3D(x, y, z1)
    # ax.scatter3D(x, y, z2)

    ax.set_xlabel('$\k_x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def main():
    plot3d(get_eigenvalues_for_plot()[0], get_eigenvalues_for_plot()[1], get_eigenvalues_for_plot()[2], get_eigenvalues_for_plot()[3])


main()


