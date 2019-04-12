import numpy as np
import matplotlib.pyplot as plt
import cmath


def define_hamiltonian(k, t, dt):
    v = t + dt
    w = t - dt
    hamiltonian = np.array([[0, v + w*np.exp(-1j*k)],
                            [v + w*np.exp(1j*k), 0]])
    return hamiltonian


def get_eigenvalues_for_plot(t, dt):
    eigenvalues_array_0 = []
    eigenvalues_array_1 = []
    k_array = []
    k = -np.pi
    k_step = 0.001
    while k < np.pi:
        eigenvalues_array_0.append(np.linalg.eigh(define_hamiltonian(k, t, dt))[0][0])
        eigenvalues_array_1.append(np.linalg.eigh(define_hamiltonian(k, t, dt))[0][1])
        k_array.append(k)
        k = k + k_step
    return k_array, eigenvalues_array_0, eigenvalues_array_1


def main():
    pass


main()

