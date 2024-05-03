import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
from ncon import ncon
import os

from hamiltonian_matrix import create_H1, create_H2, I, c_up, c_down, c_up_dagger, c_down_dagger
from infinite_dmrg import create_custom_multiply


def expand_left_block(H_L, H_Ll):
    dim_L = H_L.shape[0]
    interaction = H_Ll  # - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
    H_L_new = np.kron(H_L, I) + np.kron(np.eye(dim_L), H1) + interaction
    c_up_l_new = np.kron(np.eye(dim_L), c_up)
    c_down_l_new = np.kron(np.eye(dim_L), c_down)
    c_up_dagger_l_new = np.transpose(np.conj(c_up_l_new))
    c_down_dagger_l_new = np.transpose(np.conj(c_down_l_new))
    return H_L_new, c_up_l_new, c_down_l_new, c_up_dagger_l_new, c_down_dagger_l_new


def expand_right_block(H_R, H_rR):
    dim_R = H_R.shape[0]
    interaction = H_rR  # - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))
    H_R_new = np.kron(I, H_R) + np.kron(H1, np.eye(dim_R)) + interaction
    c_up_r_new = np.kron(c_up, np.eye(dim_R))
    c_down_r_new = np.kron(c_down, np.eye(dim_R))
    c_up_dagger_r_new = np.transpose(np.conj(c_up_r_new))
    c_down_dagger_r_new = np.transpose(np.conj(c_down_r_new))
    return H_R_new, c_up_r_new, c_down_r_new, c_up_dagger_r_new, c_down_dagger_r_new


def extend(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR, k):

    custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)
    dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]
    eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=1, which='SA')
    psi_zero = eigenvectors[:, 0]  # ground state
    gs_energy_new = eigenvalues[0] / (2 * k + 4)
    print("%d\t%.14f" % (k, gs_energy_new))

    # Density matrix construction

    dim_L = H_L.shape[0]
    dim_l = H_l.shape[0]
    dim_r = H_r.shape[0]
    dim_R = H_R.shape[0]

    psi_zero = np.reshape(psi_zero, (dim_L, dim_l, dim_r, dim_R))

    dm_left = ncon([psi_zero, np.conj(psi_zero)], [[-1, -2, 1, 2], [-3, -4, 1, 2]])
    dm_right = ncon([psi_zero, np.conj(psi_zero)], [[1, 2, -1, -2], [1, 2, -3, -4]])

    # Diagonalize dm

    dm_left = np.reshape(dm_left, (dim_L * dim_l, dim_L * dim_l))
    dm_right = np.reshape(dm_right, (dim_R * dim_r, dim_R * dim_r))
    _, eigenvectors = eigh(dm_left)
    u_left = np.fliplr(eigenvectors)
    _, eigenvectors = eigh(dm_right)
    u_right = np.fliplr(eigenvectors)

    u_left = u_left[:, :xi]
    u_right = u_right[:, :xi]

    # Expand the block
    H_L_new, c_up_l_new, c_down_l_new, c_up_dagger_l_new, c_down_dagger_l_new = expand_left_block(H_L, H_Ll)
    H_R_new, c_up_r_new, c_down_r_new, c_up_dagger_r_new, c_down_dagger_r_new = expand_right_block(H_R, H_rR)

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

    np.save(f'mem/H_L{k + 2}.npy', H_L)
    np.save(f'mem/H_R{k + 2}.npy', H_R)

    np.save(f'mem/c_up_l{k + 2}.npy', c_up_l)
    np.save(f'mem/c_down_l{k + 2}.npy', c_down_l)
    np.save(f'mem/c_up_dagger_l{k + 2}.npy', c_up_dagger_l)
    np.save(f'mem/c_down_dagger_l{k + 2}.npy', c_down_dagger_l)

    np.save(f'mem/c_up_r{k + 2}.npy', c_up_r)
    np.save(f'mem/c_down_r{k + 2}.npy', c_down_r)
    np.save(f'mem/c_up_dagger_r{k + 2}.npy', c_up_dagger_r)
    np.save(f'mem/c_down_dagger_r{k + 2}.npy', c_down_dagger_r)

    return H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR


def right_move(H_l, H_r, H_lr, left_size, right_size):

    H_L = np.load(f'mem/H_L{left_size}.npy')
    c_up_l = np.load(f'mem/c_up_l{left_size}.npy')
    c_down_l = np.load(f'mem/c_down_l{left_size}.npy')
    c_up_dagger_l = np.load(f'mem/c_up_dagger_l{left_size}.npy')
    c_down_dagger_l = np.load(f'mem/c_down_dagger_l{left_size}.npy')

    H_R = np.load(f'mem/H_R{right_size}.npy')
    c_up_r = np.load(f'mem/c_up_r{right_size}.npy')
    c_down_r = np.load(f'mem/c_down_r{right_size}.npy')
    c_up_dagger_r = np.load(f'mem/c_up_dagger_r{right_size}.npy')
    c_down_dagger_r = np.load(f'mem/c_down_dagger_r{right_size}.npy')

    H_Ll = - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
    H_rR = - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))

    custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)
    dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]
    eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=1, which='SA')
    psi_zero = eigenvectors[:, 0]  # ground state
    gs_energy = eigenvalues[0] / (left_size + right_size + 2)
    print("%s\t%.14f" % (">>>", gs_energy))

    # Density matrix construction

    dim_L = H_L.shape[0]
    dim_l = H_l.shape[0]
    dim_r = H_r.shape[0]
    dim_R = H_R.shape[0]

    psi_zero = np.reshape(psi_zero, (dim_L, dim_l, dim_r, dim_R))

    dm_left = ncon([psi_zero, np.conj(psi_zero)], [[-1, -2, 1, 2], [-3, -4, 1, 2]])

    # Diagonalize dm

    dm_left = np.reshape(dm_left, (dim_L * dim_l, dim_L * dim_l))
    _, eigenvectors = eigh(dm_left)
    u_left = np.fliplr(eigenvectors)
    u_left = u_left[:, :xi]

    # Expand the block
    H_L_new, c_up_l_new, c_down_l_new, c_up_dagger_l_new, c_down_dagger_l_new = expand_left_block(H_L, H_Ll)

    # Re-normalize

    H_L = np.transpose(np.conj(u_left)) @ H_L_new @ u_left

    c_up_l = np.transpose(np.conj(u_left)) @ c_up_l_new @ u_left
    c_down_l = np.transpose(np.conj(u_left)) @ c_down_l_new @ u_left
    c_up_dagger_l = np.transpose(np.conj(u_left)) @ c_up_dagger_l_new @ u_left
    c_down_dagger_l = np.transpose(np.conj(u_left)) @ c_down_dagger_l_new @ u_left

    np.save(f'mem/H_L{left_size + 1}.npy', H_L)

    np.save(f'mem/c_up_l{left_size + 1}.npy', c_up_l)
    np.save(f'mem/c_down_l{left_size + 1}.npy', c_down_l)
    np.save(f'mem/c_up_dagger_l{left_size + 1}.npy', c_up_dagger_l)
    np.save(f'mem/c_down_dagger_l{left_size + 1}.npy', c_down_dagger_l)


def left_move(H_l, H_r, H_lr, left_size, right_size):

    H_L = np.load(f'mem/H_L{left_size}.npy')
    c_up_l = np.load(f'mem/c_up_l{left_size}.npy')
    c_down_l = np.load(f'mem/c_down_l{left_size}.npy')
    c_up_dagger_l = np.load(f'mem/c_up_dagger_l{left_size}.npy')
    c_down_dagger_l = np.load(f'mem/c_down_dagger_l{left_size}.npy')

    H_R = np.load(f'mem/H_R{right_size}.npy')
    c_up_r = np.load(f'mem/c_up_r{right_size}.npy')
    c_down_r = np.load(f'mem/c_down_r{right_size}.npy')
    c_up_dagger_r = np.load(f'mem/c_up_dagger_r{right_size}.npy')
    c_down_dagger_r = np.load(f'mem/c_down_dagger_r{right_size}.npy')

    H_Ll = - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
    H_rR = - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))

    custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)
    dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]
    eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=1, which='SA')
    psi_zero = eigenvectors[:, 0]  # ground state
    gs_energy = eigenvalues[0] / (left_size + right_size + 2)
    print("%s\t%.14f" % ("<<<", gs_energy))

    # Density matrix construction

    dim_L = H_L.shape[0]
    dim_l = H_l.shape[0]
    dim_r = H_r.shape[0]
    dim_R = H_R.shape[0]

    psi_zero = np.reshape(psi_zero, (dim_L, dim_l, dim_r, dim_R))

    dm_right = ncon([psi_zero, np.conj(psi_zero)], [[1, 2, -1, -2], [1, 2, -3, -4]])

    # Diagonalize dm

    dm_right = np.reshape(dm_right, (dim_R * dim_r, dim_R * dim_r))
    _, eigenvectors = eigh(dm_right)
    u_right = np.fliplr(eigenvectors)
    u_right = u_right[:, :xi]

    # Expand the block
    H_R_new, c_up_r_new, c_down_r_new, c_up_dagger_r_new, c_down_dagger_r_new = expand_right_block(H_R, H_rR)

    # Re-normalize

    H_R = np.transpose(np.conj(u_right)) @ H_R_new @ u_right

    c_up_r = np.transpose(np.conj(u_right)) @ c_up_r_new @ u_right
    c_down_r = np.transpose(np.conj(u_right)) @ c_down_r_new @ u_right
    c_up_dagger_r = np.transpose(np.conj(u_right)) @ c_up_dagger_r_new @ u_right
    c_down_dagger_r = np.transpose(np.conj(u_right)) @ c_down_dagger_r_new @ u_right

    np.save(f'mem/H_R{right_size + 1}.npy', H_R)

    np.save(f'mem/c_up_r{right_size + 1}.npy', c_up_r)
    np.save(f'mem/c_down_r{right_size + 1}.npy', c_down_r)
    np.save(f'mem/c_up_dagger_r{right_size + 1}.npy', c_up_dagger_r)
    np.save(f'mem/c_down_dagger_r{right_size + 1}.npy', c_down_dagger_r)


def calculate_gs_energy(H_l, H_r, H_lr, left_size, right_size):

    H_L = np.load(f'mem/H_L{left_size}.npy')
    c_up_l = np.load(f'mem/c_up_l{left_size}.npy')
    c_down_l = np.load(f'mem/c_down_l{left_size}.npy')
    c_up_dagger_l = np.load(f'mem/c_up_dagger_l{left_size}.npy')
    c_down_dagger_l = np.load(f'mem/c_down_dagger_l{left_size}.npy')

    H_R = np.load(f'mem/H_R{right_size}.npy')
    c_up_r = np.load(f'mem/c_up_r{right_size}.npy')
    c_down_r = np.load(f'mem/c_down_r{right_size}.npy')
    c_up_dagger_r = np.load(f'mem/c_up_dagger_r{right_size}.npy')
    c_down_dagger_r = np.load(f'mem/c_down_dagger_r{right_size}.npy')

    H_Ll = - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
    H_rR = - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))

    custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)
    dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]
    eigenvalues, _ = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=1, which='SA')
    # psi_zero = eigenvectors[:, 0]  # ground state
    gs_energy = eigenvalues[0] / (left_size + right_size + 2)
    return gs_energy


if __name__ == '__main__':

    n = 3  # max system size = 2 * n + 4
    xi = 5

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

    c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l = c_up, c_down, c_up_dagger, c_down_dagger
    c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r = c_up, c_down, c_up_dagger, c_down_dagger

    if not os.path.exists('mem'):
        os.makedirs('mem')

    np.save('mem/H_L1.npy', H_L)
    np.save('mem/H_R1.npy', H_R)

    np.save('mem/c_up_l1.npy', c_up_l)
    np.save('mem/c_down_l1.npy', c_down_l)
    np.save('mem/c_up_dagger_l1.npy', c_up_dagger_l)
    np.save('mem/c_down_dagger_l1.npy', c_down_dagger_l)

    np.save('mem/c_up_r1.npy', c_up_r)
    np.save('mem/c_down_r1.npy', c_down_r)
    np.save('mem/c_up_dagger_r1.npy', c_up_dagger_r)
    np.save('mem/c_down_dagger_r1.npy', c_down_dagger_r)

    left_size = 1
    right_size = 1

    # 1) System extension
    for k in range(n):
        H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR = extend(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR, k)
        left_size += 1
        right_size += 1
        print(left_size, right_size)

    i = 0

    gs_energy = 0
    gs_energy_new = -1

    # 2) Sweeping
    # while abs(gs_energy - gs_energy_new) > 1.e-14:
    for _ in range(100):
        for _ in range(n):
            right_move(H_l, H_r, H_lr, left_size, right_size)
            left_size += 1
            right_size -= 1
            # print(left_size, right_size)
        for _ in range(2 * n):
            left_move(H_l, H_r, H_lr, left_size, right_size)
            left_size -= 1
            right_size += 1
            # print(left_size, right_size)
        for _ in range(n):
            right_move(H_l, H_r, H_lr, left_size, right_size)
            left_size += 1
            right_size -= 1
            # print(left_size, right_size)
        gs_energy = gs_energy_new
        gs_energy_new = calculate_gs_energy(H_l, H_r, H_lr, n + 1, n + 1)
        print("%d\t%.14f" % (i, gs_energy_new))
        i += 1
