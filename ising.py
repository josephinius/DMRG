import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
from ncon import ncon

from arnoldi_diag_demo import create_custom_multiply

# Spin-1/2 operators

sx = np.array([
    [0, 1],
    [1, 0]
])

sz = np.array([
    [1, 0],
    [0, -1]
])

"""
sy = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)
"""

I = np.eye(2)


if __name__ == '__main__':

    xi = 16
    h = 1.e-0

    # sx_l, sy_l, sz_l = sx, sy, sz
    # sx_r, sy_r, sz_r = sx, sy, sz

    sx_l, sz_l = sx, sz
    sx_r, sz_r = sx, sz

    H1 = - h * sx
    # H2 = - np.kron(sx, sx) - np.kron(sy, sy) - np.kron(sz, sz)
    H2 = - np.kron(sz, sz)

    H_L = H1
    H_l = H1
    H_r = H1
    H_R = H1

    H_Ll = H2
    H_lr = H2
    H_rR = H2

    system_size = 4  # initial system size

    # file_name = f"data_heisenberg_xi{xi}.txt"
    file_name = f"data_tfim_xi{xi}.txt"

    with open(file_name, 'w') as f:
        f.write('# iter\tenergy\n')

    for iter_count in range(1000):

        # print('iter_count', iter_count)

        custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)

        dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]

        eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=1, which='SA')
        # print(eigenvalues / system_size)
        # print(iter_count, eigenvalues[0] / system_size)  # ground state energy per site
        gs_energy = eigenvalues[0] / system_size
        print("%d\t%.14f" % (iter_count, gs_energy))
        with open(file_name, 'a') as f:
            f.write('%d\t%.15f\n' % (iter_count, gs_energy))

        psi_zero = eigenvectors[:, 0]  # ground state
        # psi_zero[np.abs(psi_zero) < 1.e-16] = 0
        # print(psi_zero)

        # Density matrix construction

        dim_L = H_L.shape[0]
        dim_l = H_l.shape[0]
        dim_r = H_r.shape[0]
        dim_R = H_R.shape[0]

        psi_zero = np.reshape(psi_zero, (dim_L, dim_l, dim_r, dim_R))
        # print(psi_zero.shape)

        dm_left = ncon([psi_zero, np.conj(psi_zero)], [[-1, -2, 1, 2], [-3, -4, 1, 2]])
        dm_right = ncon([psi_zero, np.conj(psi_zero)], [[1, 2, -1, -2], [1, 2, -3, -4]])

        # Diagonalize dm

        dm_left = np.reshape(dm_left, (dim_L * dim_l, dim_L * dim_l))
        dm_right = np.reshape(dm_right, (dim_R * dim_r, dim_R * dim_r))
        # print(dm_left.shape)
        eigenvalues, eigenvectors = eigh(dm_left)
        # eigenvalues, eigenvectors = eigsh(dm_left, k=xi, which='LA')
        # print('eigenvalues: ', eigenvalues)
        u_left = np.fliplr(eigenvectors)

        eigenvalues, eigenvectors = eigh(dm_right)
        # eigenvalues, eigenvectors = eigsh(dm_right, k=xi, which='LA')
        u_right = np.fliplr(eigenvectors)

        u_left = u_left[:, :xi]
        u_right = u_right[:, :xi]

        # print(u_left.shape)

        # Expand the block

        # interaction = - np.kron(sx_l, sx) - np.kron(sy_l, sy) - np.kron(sz_l, sz)
        interaction = - np.kron(sz_l, sz)
        H_L_new = np.kron(H_L, I) + np.kron(np.eye(dim_L), H1) + interaction
        # sx_l_new = np.kron(np.eye(dim_L), sx)
        # sy_l_new = np.kron(np.eye(dim_L), sy)
        sz_l_new = np.kron(np.eye(dim_L), sz)

        # interaction = - np.kron(sx, sx_r) - np.kron(sy, sy_r) - np.kron(sz, sz_r)
        interaction = - np.kron(sz, sz_r)
        H_R_new = np.kron(I, H_R) + np.kron(H1, np.eye(dim_R)) + interaction
        # sx_r_new = np.kron(sx, np.eye(dim_R))
        # sy_r_new = np.kron(sy, np.eye(dim_R))
        sz_r_new = np.kron(sz, np.eye(dim_R))

        # Re-normalize

        H_L = np.transpose(np.conj(u_left)) @ H_L_new @ u_left
        H_R = np.transpose(np.conj(u_right)) @ H_R_new @ u_right

        # sx_l = np.transpose(np.conj(u_left)) @ sx_l_new @ u_left
        # sy_l = np.transpose(np.conj(u_left)) @ sy_l_new @ u_left
        sz_l = np.transpose(np.conj(u_left)) @ sz_l_new @ u_left

        # sx_r = np.transpose(np.conj(u_right)) @ sx_r_new @ u_right
        # sy_r = np.transpose(np.conj(u_right)) @ sy_r_new @ u_right
        sz_r = np.transpose(np.conj(u_right)) @ sz_r_new @ u_right

        # H_Ll = - np.kron(sx_l, sx) - np.kron(sy_l, sy) - np.kron(sz_l, sz)
        H_Ll = - np.kron(sz_l, sz)
        # H_rR = - np.kron(sx, sx_r) - np.kron(sy, sy_r) - np.kron(sz, sz_r)
        H_rR = - np.kron(sz, sz_r)

        system_size += 2
