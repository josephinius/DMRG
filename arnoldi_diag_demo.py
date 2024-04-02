import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from ncon import ncon

from hamiltonian_matrix import create_H1, create_H2


if __name__ == '__main__':

    t = 1.  # hopping parameter
    mu = 2 * t  # chemical potential
    U = 0.  # on-site Coulomb interaction energy

    H1 = create_H1(t, mu, U)
    H2 = create_H2(t)

    H_L = H1
    H_l = H1
    H_r = H1
    H_R = H1

    H_Ll = H2
    H_lr = H2
    H_rR = H2

    # a = np.diag([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(a)

    def custom_multiply(vec):
        dim_L = H_L.shape[0]
        dim_l = H_l.shape[0]
        dim_r = H_r.shape[0]
        dim_R = H_R.shape[0]
        vec = np.reshape(vec, (dim_L, dim_l, dim_r, dim_R))
        m1 = ncon([H_L, vec], [[-1, 1], [1, -2, -3, -4]])
        m2 = ncon([H_l, vec], [[-2, 1], [-1, 1, -3, -4]])
        m3 = ncon([H_r, vec], [[-3, 1], [-1, -2, 1, -4]])
        m4 = ncon([H_R, vec], [[-4, 1], [-1, -2, -3, 1]])
        H_Llp = np.reshape(H_Ll, (dim_L, dim_l, dim_L, dim_l))
        m5 = ncon([H_Llp, vec], [[-1, -2, 1, 2], [1, 2, -3, -4]])
        H_lrp = np.reshape(H_lr, (dim_l, dim_r, dim_l, dim_r))
        m6 = ncon([H_lrp, vec], [[-2, -3, 1, 2], [-1, 1, 2, -4]])
        H_rRp = np.reshape(H_rR, (dim_r, dim_R, dim_r, dim_R))
        m7 = ncon([H_rRp, vec], [[-3, -4, 1, 2], [-1, -2, 1, 2]])
        res = m1 + m2 + m3 + m4 + m5 + m6 + m7
        return np.reshape(res, (-1))


    dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]

    eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=6, which='SA')
    print(eigenvalues)
    psi_zero = eigenvectors[:, 0]
    psi_zero[np.abs(psi_zero) < 1.e-14] = 0
    print(psi_zero)
