import numpy as np
from scipy.linalg import eigh
# from scipy.sparse.linalg import eigsh
# import ncon


I = np.eye(4)

# annihilation operator for spin up
c_up = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

# annihilation operator for spin down
c_down = np.matrix([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

c_up_dagger = np.transpose(np.conj(c_up))  # creation operator for spin up
c_down_dagger = np.transpose(np.conj(c_down))  # creation operator for spin down

n_up = c_up_dagger @ c_up  # number operator for spin up
n_down = c_down_dagger @ c_down  # number operator for spin down


if __name__ == '__main__':
    t = 1.  # hopping parameter
    mu = 2 * t  # chemical potential
    U = 0.  # on-site Coulomb interaction energy

    # print('n_up:\n', n_up)
    # print('n_down:\n', n_down)

    H1 = (2 * t - mu) * (n_up + n_down) + U * (n_up @ n_down)
    H2 = - t * (np.kron(c_up_dagger, c_up) + np.kron(c_up, c_up_dagger) +
                np.kron(c_down_dagger, c_down) + np.kron(c_down, c_down_dagger))

    # print('H1.shape', H1.shape)
    # print(H1)
    # print('H2.shape', H2.shape)
    # print(H2)

    H2_full = H2 + np.kron(H1, I) + np.kron(I, H1)
    print('H2_full.shape', H2_full.shape)

    eigenvalues, eigenvectors = eigh(H2_full)
    print(eigenvalues)
    psi_zero = eigenvectors[:, 0]  # ground state of two-site system
    print(psi_zero)

    N2 = np.kron(n_up, I) + np.kron(n_down, I) + np.kron(I, n_up) + np.kron(I, n_down)

    left = np.reshape(psi_zero, (1, 16))
    right = np.reshape(psi_zero, (16, 1))
    p = np.dot(left, N2 @ right)  # number of particles
    print('p: ', p)

    S2 = np.kron(n_up, I) - np.kron(n_down, I) + np.kron(I, n_up) - np.kron(I, n_down)

    s = np.dot(left, S2 @ right)  # total spin in z direction
    print('s: ', s)

    H4 = np.kron(np.kron(np.kron(H1, I), I), I) + np.kron(np.kron(np.kron(I, H1), I), I) + \
         np.kron(np.kron(np.kron(I, I), H1), I) + np.kron(np.kron(np.kron(I, I), I), H1) + \
         np.kron(np.kron(H2, I), I) + np.kron(np.kron(I, H2), I) + np.kron(np.kron(I, I), H2)

    eigenvalues, eigenvectors = eigh(H4)
    print('eigenvalues: ', eigenvalues[:10])
    psi_zero = eigenvectors[:, 0]

    N4 = np.kron(np.kron(n_up, I), np.kron(I, I)) + np.kron(np.kron(I, n_up), np.kron(I, I)) + \
         np.kron(np.kron(I, I), np.kron(n_up, I)) + np.kron(np.kron(I, I), np.kron(I, n_up)) + \
         np.kron(np.kron(n_down, I), np.kron(I, I)) + np.kron(np.kron(I, n_down), np.kron(I, I)) + \
         np.kron(np.kron(I, I), np.kron(n_down, I)) + np.kron(np.kron(I, I), np.kron(I, n_down))

    left = np.reshape(psi_zero, (1, -1))
    right = np.reshape(psi_zero, (-1, 1))
    p = np.dot(left, N4 @ right)
    print('p: ', p)

    S4 = np.kron(np.kron(n_up, I), np.kron(I, I)) + np.kron(np.kron(I, n_up), np.kron(I, I)) + \
         np.kron(np.kron(I, I), np.kron(n_up, I)) + np.kron(np.kron(I, I), np.kron(I, n_up)) - \
         np.kron(np.kron(n_down, I), np.kron(I, I)) - np.kron(np.kron(I, n_down), np.kron(I, I)) - \
         np.kron(np.kron(I, I), np.kron(n_down, I)) - np.kron(np.kron(I, I), np.kron(I, n_down))

    s = np.dot(left, S4 @ right)
    print('s: ', s)
