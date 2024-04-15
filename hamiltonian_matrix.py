import numpy as np
from scipy.linalg import eigh

"""

 |0> = (1, 0, 0, 0)  # no particle
 |1> = (0, 1, 0, 0)  # electron spin up
 |2> = (0, 0, 1, 0)  # electron spin down
 |3> = (0, 0, 0, 1)  # two electrons (up & down)

"""


I = np.eye(4)

# annihilation operator for spin up
c_up = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])

# annihilation operator for spin down
c_down = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

c_up_dagger = np.transpose(np.conj(c_up))  # creation operator for spin up
c_down_dagger = np.transpose(np.conj(c_down))  # creation operator for spin down

n_up = c_up_dagger @ c_up  # number operator for spin up
n_down = c_down_dagger @ c_down  # number operator for spin down


def create_H1(t, mu, U):
    H1 = (2 * t - mu) * (n_up + n_down) + U * (n_up @ n_down)
    return H1


def create_H2(t):
    H2 = - t * (np.kron(c_up_dagger, c_up) + np.kron(c_up, c_up_dagger) +
                np.kron(c_down_dagger, c_down) + np.kron(c_down, c_down_dagger))
    return H2


if __name__ == '__main__':
    t = 1.  # hopping parameter
    mu = 2 * t  # chemical potential
    U = 0.  # on-site Coulomb interaction energy

    # print('n_up:\n', n_up)
    # print('n_down:\n', n_down)

    H1 = create_H1(t, mu, U)
    H2 = create_H2(t)

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
    print('eigenvalues: ', eigenvalues[:6])
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

    psi_zero[np.abs(psi_zero) < 1.e-14] = 0
    # print(psi_zero)

    n = 0  # Number of particles counted "manually"
    astate = []
    particles = {0: 0, 1: 1, 2: 1, 3: 2}
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    i = 64 * i1 + 16 * i2 + 4 * i3 + i4
                    if abs(psi_zero[i]) > 1.e-14:
                        astate.append([psi_zero[i], (i1, i2, i3, i4)])
                    n += (psi_zero[i] ** 2) * (particles[i1] + particles[i2] + particles[i3] + particles[i4])

    print('n: ', n)

    # Creating the spin representation of the state in a different way

    astate2 = []

    for i in range(4 ** 4):
        ii = []
        for _ in range(4):
            ii.append(i % 4)
            i //= 4
        ii = reversed(ii)
        if abs(psi_zero[i]) > 1.e-14:
            astate2.append([psi_zero[i], tuple(ii)])

    # for x in astate:
    #    print("%.8f - %d%d%d%d" % (x[0], *x[1]))

    for x1, x2 in zip(astate, astate2):
        if abs(x1[0] - x2[0]) > 1.e-14 or x1[1] != x2[1]:
            print(x1[0], x2[0])
            print(x1[1], x2[1])
            print("not same!")
            break

    print(sum(psi_zero ** 2))  # checking normalisation

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

    print(H6.shape)

    eigenvalues, eigenvectors = eigh(H6)
    print('eigenvalues: ', eigenvalues[:6] / 6)
    print(eigenvalues[0] / 6)  # -1.164653069144978

    """
    # H8 takes too much of memory... 
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
    print('eigenvalues: ', eigenvalues[:6] / 8)
    print(eigenvalues[0] / 8)
    """
