
import numpy as np
import matplotlib.pyplot as plt
import cmath
from mpl_toolkits.mplot3d import Axes3D


t = 1
lamb = 0
V = 1

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


def get_eigenvalues_for_plot():

    a1 = 0.5 * np.array([-3**0.5, 3])
    a2 = 0.5 * np.array([3**0.5, 3])

    eigenvalues_array_0 = []
    eigenvalues_array_1 = []

    kx_array = []
    ky_array = []

    kx = -np.pi
    ky = -np.pi
    k_step = 0.01

    for kx in np.arange(-np.pi, np.pi, k_step):
        k1 = np.dot(np.array([kx, ky]), a1)
        for ky in np.arange(-np.pi, np.pi, k_step):
            k2 = np.dot(np.array([kx, ky]), a2)

            eigenvalues_array_0.append(np.linalg.eigh(define_hamiltonian(k1, k2))[0][0])
            eigenvalues_array_1.append(np.linalg.eigh(define_hamiltonian(k1, k2))[0][1])

            ky_array.append(ky)
            kx_array.append(kx)

    return kx_array, ky_array, eigenvalues_array_0, eigenvalues_array_1


def plot3d(X, Y, z1, z2):

    # x, y = np.meshgrid(X, Y)
    # print(X)
    # print(Y)
    # print(z1)
    # print(z2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print(len(Y), len(X))

    ax.scatter3D(X, Y, z2)
    ax.set_xlabel('$\k_x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def main():
    plot3d(get_eigenvalues_for_plot()[0], get_eigenvalues_for_plot()[1], get_eigenvalues_for_plot()[2], get_eigenvalues_for_plot()[3])


main()


