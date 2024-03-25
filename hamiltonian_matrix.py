import numpy as np
from scipy.sparse.linalg import eigsh
# import ncon


id = np.eye(4)

c_up = np.matrix([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

c_down = np.matrix([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

c_up_dagger = np.transpose(np.conj(c_up))
c_down_dagger = np.transpose(np.conj(c_down))

n_up = c_up_dagger @ c_up
n_down = c_down_dagger @ c_down


if __name__ == '__main__':
    t = 1.
    mu = 0.5
    U = 0.

    # print('n_up:\n', n_up)
    # print('n_down:\n', n_down)

    H1 = (2 * t - mu) * (n_up + n_down) + U * (n_up @ n_down)
    H2 = - t * (np.kron(c_up_dagger, c_up) + np.kron(c_up_dagger, c_up) +
                np.kron(c_down_dagger, c_down) + np.kron(c_down_dagger, c_down))

    print('H1.shape', H1.shape)
    print('H2.shape', H2.shape)

    H4 = np.kron(np.kron(np.kron(H1, id), id), id) + np.kron(np.kron(np.kron(id, H1), id), id) + \
         np.kron(np.kron(np.kron(id, id), H1), id) + np.kron(np.kron(np.kron(id, id), id), H1) + \
         np.kron(np.kron(H2, id), id) + np.kron(np.kron(id, H2), id) + np.kron(np.kron(id, id), H2)

    print('H4.shape', H4.shape)

    eigenvalues, eigenvectors = eigsh(H4, k=10)
    print('eigenvalues: ', eigenvalues)
