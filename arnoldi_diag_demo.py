import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
from ncon import ncon

from hamiltonian_matrix import create_H1, create_H2, I, c_up, c_down, c_up_dagger, c_down_dagger


def create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR):
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
    return custom_multiply


if __name__ == '__main__':

    xi = 100

    t = 1.  # hopping parameter
    mu = 2 * t  # chemical potential
    U = 0.  # on-site Coulomb interaction energy

    c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l = c_up, c_down, c_up_dagger, c_down_dagger
    c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r = c_up, c_down, c_up_dagger, c_down_dagger

    H1 = create_H1(t, mu, U)
    H2 = create_H2(t)

    H_L = H1
    H_l = H1
    H_r = H1
    H_R = H1

    H_Ll = H2
    H_lr = H2
    H_rR = H2

    system_size = 4  # initial system size

    for iter_count in range(1000):

        custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)

        dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]

        eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=6, which='SA')
        # print(eigenvalues / system_size)
        print(iter_count, eigenvalues[0] / system_size)
        psi_zero = eigenvectors[:, 0]
        # psi_zero[np.abs(psi_zero) < 1.e-14] = 0
        # print(psi_zero)

        # Density matrix construction

        dim_L = H_L.shape[0]
        dim_l = H_l.shape[0]
        dim_r = H_r.shape[0]
        dim_R = H_R.shape[0]

        psi_zero = np.reshape(psi_zero, (dim_L, dim_l, dim_r, dim_R))
        # print(psi_zero.shape)

        dm_left = ncon([psi_zero, np.conj(psi_zero)], [[-1, -2, 1, 2], [-3, -4, 1, 2]])
        dm_right = ncon([psi_zero, np.conj(psi_zero)], [[2, 1, -2, -1], [2, 1, -4, -3]])

        # Diagonalize dm

        dm_left = np.reshape(dm_left, (dim_L * dim_l, dim_L * dim_l))
        dm_right = np.reshape(dm_right, (dim_R * dim_r, dim_R * dim_r))
        # print(dm_left.shape)
        # eigenvalues, eigenvectors = eigh(dm_left)
        # print(eigenvalues)
        eigenvalues, eigenvectors = eigsh(dm_left, k=xi, which='LA')
        # print('eigenvalues: ', eigenvalues)
        u_left = np.fliplr(eigenvectors)

        # _, eigenvectors = eigh(dm_right)
        eigenvalues, eigenvectors = eigsh(dm_right, k=xi, which='LA')
        u_right = np.fliplr(eigenvectors)

        u_left = u_left[:, :xi]
        u_right = u_right[:, :xi]

        # print(u_left.shape)

        # Expand the block

        interaction = - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
        H_L_new = np.kron(H_L, I) + np.kron(np.eye(dim_L), H1) + interaction
        c_up_l_new = np.kron(np.eye(dim_L), c_up)
        c_down_l_new = np.kron(np.eye(dim_L), c_down)
        c_up_dagger_l_new = np.transpose(np.conj(c_up_l_new))
        c_down_dagger_l_new = np.transpose(np.conj(c_down_l_new))

        interaction = - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))
        H_R_new = np.kron(I, H_R) + np.kron(H1, np.eye(dim_R)) + interaction
        c_up_r_new = np.kron(c_up, np.eye(dim_R))
        c_down_r_new = np.kron(c_down, np.eye(dim_R))
        c_up_dagger_r_new = np.transpose(np.conj(c_up_r_new))
        c_down_dagger_r_new = np.transpose(np.conj(c_down_r_new))

        # Re-normalize

        H_L = np.transpose(np.conj(u_left)) @ H_L_new @ u_left
        H_R = np.transpose(np.conj(u_right)) @ H_R_new @ u_right

        c_up_l = np.transpose(np.conj(u_left)) @ c_up_l_new @ u_left
        c_down_l = np.transpose(np.conj(u_left)) @ c_down_l_new @ u_left
        c_up_dagger_l = np.transpose(np.conj(u_left)) @ c_up_dagger_l_new @ u_left
        c_down_dagger_l = np.transpose(np.conj(u_left)) @ c_down_dagger_l_new @ u_left

        c_up_r = np.transpose(np.conj(u_right)) @ c_up_r_new @ u_right
        c_down_r = np.transpose(np.conj(u_right)) @ c_down_r_new @ u_right
        c_up_dagger_r = np.transpose(np.conj(u_right)) @ c_up_dagger_r_new @ u_right
        c_down_dagger_r = np.transpose(np.conj(u_right)) @ c_down_dagger_r_new @ u_right

        H_Ll = - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
        H_rR = - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))

        system_size += 2
