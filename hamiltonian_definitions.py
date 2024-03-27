import numpy as np
# from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh


I = np.eye(4)

# TODO: check all signs in h2
h2 = np.matrix([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
    [0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])


def create_H2(t):
    return t * h2


def create_H1(t, mu, U):
    res = np.matrix([
        [0, 0, 0, 0],
        [0, 2 * t - mu, 0, 0],
        [0, 0, 2 * t - mu, 0],
        [0, 0, 0, 2 * (2 * t - mu) + U]
    ])
    return res


if __name__ == '__main__':

    t = 1.
    mu = 2 * t
    U = 0.

    H1 = create_H1(t, mu, U)
    H2 = create_H2(t)

    # print('H1.shape', H1.shape)
    # print(H1)
    # print('H2.shape', H2.shape)
    # print(H2)

    H4 = np.kron(np.kron(np.kron(H1, I), I), I) + np.kron(np.kron(np.kron(I, H1), I), I) + \
         np.kron(np.kron(np.kron(I, I), H1), I) + np.kron(np.kron(np.kron(I, I), I), H1) + \
         np.kron(np.kron(H2, I), I) + np.kron(np.kron(I, H2), I) + np.kron(np.kron(I, I), H2)

    # print('H4.shape', H4.shape)

    # eigenvalues, eigenvectors = eigsh(H4, k=10, which='SM')
    # eigenvalues, eigenvectors = eigsh(H4, k=6)
    # eigenvalues, eigenvectors = eigh(H4)
    # print('eigenvalues: ', eigenvalues)
    # print(eigenvectors.shape)
    # print(eigenvectors[:, -1])

    # H2_full = H2 + np.kron(H1, I) + np.kron(I, H1)
    # print('H2_full.shape', H2_full.shape)

    # eigenvalues, eigenvectors = eigh(H2_full)
    # print(eigenvalues)
    # psi_zero = eigenvectors[:, 0]
    # print(psi_zero)
