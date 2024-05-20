import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh
from ncon import ncon
import os

from hamiltonian_matrix import create_H1, create_H2, I, c_up, c_down, c_up_dagger, c_down_dagger
from infinite_dmrg import create_custom_multiply


def expand_left_block(H_L, H_Ll):
    """
    Expands the left block Hamiltonian.
    """
    dim_L = H_L.shape[0]
    interaction = H_Ll  # - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
    H_L_new = np.kron(H_L, I) + np.kron(np.eye(dim_L), H1) + interaction
    c_up_l_new = np.kron(np.eye(dim_L), c_up)
    c_down_l_new = np.kron(np.eye(dim_L), c_down)
    c_up_dagger_l_new = np.transpose(np.conj(c_up_l_new))
    c_down_dagger_l_new = np.transpose(np.conj(c_down_l_new))
    return H_L_new, c_up_l_new, c_down_l_new, c_up_dagger_l_new, c_down_dagger_l_new


def expand_right_block(H_R, H_rR):
    """
    Expands the right block Hamiltonian.
    """
    dim_R = H_R.shape[0]
    interaction = H_rR  # - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))
    H_R_new = np.kron(I, H_R) + np.kron(H1, np.eye(dim_R)) + interaction
    c_up_r_new = np.kron(c_up, np.eye(dim_R))
    c_down_r_new = np.kron(c_down, np.eye(dim_R))
    c_up_dagger_r_new = np.transpose(np.conj(c_up_r_new))
    c_down_dagger_r_new = np.transpose(np.conj(c_down_r_new))
    return H_R_new, c_up_r_new, c_down_r_new, c_up_dagger_r_new, c_down_dagger_r_new


def save_half(H, c_up, c_down, c_up_dagger, c_down_dagger, size, which):  # which is 'l' or 'r'
    """
    Saves the Hamiltonian and operators to disk.
    """
    np.save(f'mem/H_{which.upper()}{size}.npy', H)
    np.save(f'mem/c_up_{which}{size}.npy', c_up)
    np.save(f'mem/c_down_{which}{size}.npy', c_down)
    np.save(f'mem/c_up_dagger_{which}{size}.npy', c_up_dagger)
    np.save(f'mem/c_down_dagger_{which}{size}.npy', c_down_dagger)


def extend(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR, k):
    """
    Extends the system by one site on both sides.
    """

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

    # Save H-s and c-s
    save_half(H_L, c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l, k + 2, 'l')
    save_half(H_R, c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r, k + 2, 'r')

    return H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR


def load_half(size, which):  # which is 'l' or 'r'
    """
    Loads the Hamiltonian and operators from disk.
    """
    H = np.load(f'mem/H_{which.upper()}{size}.npy')
    c_up = np.load(f'mem/c_up_{which}{size}.npy')
    c_down = np.load(f'mem/c_down_{which}{size}.npy')
    c_up_dagger = np.load(f'mem/c_up_dagger_{which}{size}.npy')
    c_down_dagger = np.load(f'mem/c_down_dagger_{which}{size}.npy')
    return H, c_up, c_down, c_up_dagger, c_down_dagger


def right_move(H_l, H_r, H_lr, left_size, right_size):
    """
    Extends the left part of the system by one site.
    """
    H_L, c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l = load_half(left_size, 'l')
    H_R, c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r = load_half(right_size, 'r')

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

    save_half(H_L, c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l, left_size + 1, 'l')


def left_move(H_l, H_r, H_lr, left_size, right_size):
    """
    Extends the right part of the system by one site.
    """
    H_L, c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l = load_half(left_size, 'l')
    H_R, c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r = load_half(right_size, 'r')

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

    save_half(H_R, c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r, right_size + 1, 'r')


def calculate_gs_energy(H_l, H_r, H_lr, left_size, right_size):
    """
    Calculates the ground state energy.
    """
    H_L, c_up_l, c_down_l, c_up_dagger_l, c_down_dagger_l = load_half(left_size, 'l')
    H_R, c_up_r, c_down_r, c_up_dagger_r, c_down_dagger_r = load_half(right_size, 'r')

    H_Ll = - t * (np.kron(c_up_dagger_l, c_up) + np.kron(c_up_l, c_up_dagger) + np.kron(c_down_dagger_l, c_down) + np.kron(c_down_l, c_down_dagger))
    H_rR = - t * (np.kron(c_up_dagger, c_up_r) + np.kron(c_up, c_up_dagger_r) + np.kron(c_down_dagger, c_down_r) + np.kron(c_down, c_down_dagger_r))

    custom_multiply = create_custom_multiply(H_L, H_l, H_r, H_R, H_Ll, H_lr, H_rR)
    dim_H = H_L.shape[0] * H_l.shape[0] * H_r.shape[0] * H_R.shape[0]
    eigenvalues, eigenvectors = eigsh(LinearOperator((dim_H, dim_H), matvec=custom_multiply), k=1, which='SA')
    psi_zero = eigenvectors[:, 0]  # ground state
    gs_energy = eigenvalues[0] / (left_size + right_size + 2)
    return gs_energy, psi_zero


def occupancy(psi, H_l, H_r, left_size, right_size):
    """
    Calculates the occupancy in the center.
    """
    H_L = np.load(f'mem/H_L{left_size}.npy')
    H_R = np.load(f'mem/H_R{right_size}.npy')
    dim_L = H_L.shape[0]
    dim_l = H_l.shape[0]
    dim_r = H_r.shape[0]
    dim_R = H_R.shape[0]

    psi = np.reshape(psi, (dim_L, dim_l, dim_r, dim_R))

    n_up = c_up_dagger @ c_up
    n_down = c_down_dagger @ c_down

    n1_up = ncon([psi, np.conj(psi), n_up], [[2, 1, 4, 5], [2, 3, 4, 5], [1, 3]])
    n1_down = ncon([psi, np.conj(psi), n_down], [[2, 1, 4, 5], [2, 3, 4, 5], [1, 3]])

    n2_up = ncon([psi, np.conj(psi), n_up], [[2, 3, 1, 5], [2, 3, 4, 5], [1, 4]])
    n2_down = ncon([psi, np.conj(psi), n_down], [[2, 3, 1, 5], [2, 3, 4, 5], [1, 4]])

    return n1_up, n1_down, n2_up, n2_down


if __name__ == '__main__':

    n = 2  # max system size N = 2 * n + 4
    xi = 20

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

    file_name_center = f"energy_sweeping_N{2 * n + 4}_xi{xi}.txt"
    with open(file_name_center, 'w') as f:
        f.write('# iter, energy\n')

    gs_energy, _ = calculate_gs_energy(H_l, H_r, H_lr, n + 1, n + 1)
    print("%.14f" % gs_energy)

    with open(file_name_center, 'a') as f:
        f.write('%d\t%.15f\n' % (0, gs_energy))

    num_of_sweeps = 0

    # 2) Sweeping
    while abs(gs_energy - gs_energy_new) > 1.e-14:
        # for num_of_sweeps in range(10):
        for _ in range(n):
            right_move(H_l, H_r, H_lr, left_size, right_size)
            left_size += 1
            right_size -= 1
            print(left_size, right_size)
        for _ in range(2 * n):
            left_move(H_l, H_r, H_lr, left_size, right_size)
            left_size -= 1
            right_size += 1
            print(left_size, right_size)
        for _ in range(n):
            right_move(H_l, H_r, H_lr, left_size, right_size)
            left_size += 1
            right_size -= 1
            print(left_size, right_size)
        gs_energy = gs_energy_new
        gs_energy_new, psi_zero = calculate_gs_energy(H_l, H_r, H_lr, n + 1, n + 1)
        print("%d\t%.14f" % (i, gs_energy_new))
        n1_up, n1_down, n2_up, n2_down = occupancy(psi_zero, H_l, H_r, n + 1, n + 1)
        print(n1_up, n1_down, n2_up, n2_down)
        with open(file_name_center, 'a') as f:
            f.write('%d\t%.15f\n' % (num_of_sweeps + 1, gs_energy_new))
        i += 1

        """
        file_name = f"energy_profile_N{2 * n + 4}_xi{xi}_{num_of_sweeps + 1}.txt"
        with open(file_name, 'w') as f:
            f.write('# loc, energy\n')

        for left in range(1, 2 * n + 2):
            right = 2 * n + 2 - left
            gs_energy_cut, _ = calculate_gs_energy(H_l, H_r, H_lr, left, right)
            with open(file_name, 'a') as f:
                f.write('%d\t%.15f\n' % (left, gs_energy_cut))
        """

        num_of_sweeps += 1
