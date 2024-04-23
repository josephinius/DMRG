import numpy as np
from scipy.linalg import eigh

sx = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

sz = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)

I = np.eye(2)


if __name__ == '__main__':

    h = 0.e-14

    H1 = - h * sx
    H2 = - np.kron(sz, sz)

    H4 = np.kron(np.kron(np.kron(H1, I), I), I) + np.kron(np.kron(np.kron(I, H1), I), I) + \
         np.kron(np.kron(np.kron(I, I), H1), I) + np.kron(np.kron(np.kron(I, I), I), H1) + \
         np.kron(np.kron(H2, I), I) + np.kron(np.kron(I, H2), I) + np.kron(np.kron(I, I), H2)

    eigenvalues, eigenvectors = eigh(H4)
    print('4: ', eigenvalues[0] / 4)

    H6 = np.kron(np.kron(np.kron(np.kron(np.kron(H1, I), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(I, H1), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(I, I), H1), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), H1), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), H1), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), H1) + \
         np.kron(np.kron(np.kron(np.kron(H2, I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(I, H2), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(I, I), H2), I), I) + \
         np.kron(np.kron(np.kron(np.kron(I, I), I), H2), I) + \
         np.kron(np.kron(np.kron(np.kron(I, I), I), I), H2)

    eigenvalues, eigenvectors = eigh(H6)
    print('6: ', eigenvalues[0] / 6)

    H8 = np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(H1, I), I), I), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, H1), I), I), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), H1), I), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), H1), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), H1), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), H1), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), I), H1), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), I), I), H1) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(H2, I), I), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, H2), I), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), H2), I), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), H2), I), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), H2), I), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), H2), I) + \
         np.kron(np.kron(np.kron(np.kron(np.kron(np.kron(I, I), I), I), I), I), H2)
    print(H8.shape)

    eigenvalues, eigenvectors = eigh(H8)
    print('8: ', eigenvalues[0] / 8)
