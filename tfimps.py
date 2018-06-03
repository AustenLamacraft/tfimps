import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers
import tensorflow as tf


#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

class Tfimps:
    """
    Infinite Matrix Product State class.
    """

    # TODO Allow for two-site unit cell and average energy between A_1 A_2 and A_2 A_1 ordering
    def __init__(self, phys_d, bond_d, A_matrices=None, symmetrize=True, hamiltonian=None, r_prec=1e-14):
        """
        :param phys_d: Physical dimension of the state e.g. 2 for spin-1/2 systems.
        :param bond_d: Bond dimension, the size of the A matrices.
        :param A_matrices: Square matrices of size `bond_d` forming the Matrix Product State.
        :param symmetrize: Boolean indicating A matrices are symmetrized.
        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two site Hamiltonian
        """

        self._session = tf.Session()

        self.r_prec = r_prec
        self.phys_d = phys_d
        self.bond_d = bond_d
        self.hamiltonian = hamiltonian

        self.mps_manifold = pymanopt.manifolds.Stiefel(phys_d * bond_d, bond_d)

        # Define the A
        if A_matrices is None:
            A_init = tf.reshape(self.mps_manifold.rand(), [phys_d, bond_d, bond_d])
            # A_init = self._symmetrize(np.random.rand(phys_d, bond_d, bond_d))

        else:
            A_init = A_matrices

        # Create Stiefel from the A
        Stiefel_init = tf.reshape(A_init, [self.phys_d * self.bond_d, self.bond_d])

        # Define the variational tensor variable Stiefel
        # self.A = tf.get_variable("A_matrices", initializer=A_init, trainable=True)
        self.Stiefel = tf.get_variable("Stiefel_matrix", initializer=Stiefel_init, trainable=True,dtype=tf.float64)
        self.A = tf.reshape(self.Stiefel, [self.phys_d, self.bond_d, self.bond_d])

        if symmetrize:
            self.A = self._symmetrize(self.A)

        self._transfer_matrix = None
        self._right_eigenvector = None

        self._all_eig = tf.self_adjoint_eig(self.transfer_matrix)
        self._dominant_eig = None

        self._variational_energy = None

        if hamiltonian is not None:
            if symmetrize:
                self.variational_energy = self._add_variational_energy_symmetric_mps(hamiltonian)
            else:
                self.variational_energy = self._add_variational_energy_left_canonical_mps(hamiltonian)

    def correlator(self, operator, range):
        """
        Evaluate the correlation function of `operator` up to `range` sites.

        :param operator: Tensor of shape [phys_d, phys_d] giving single site operator.
        :param range: Maximum separation at which correlations required
        :return: Correlation function
        """
        dom_eigval, dom_eigvec = self.dominant_eig
        dom_eigmat = tf.reshape(dom_eigvec, [self.bond_d, self.bond_d])
        #
        eigval, eigvec = self._all_eig
        eigtens = tf.reshape(tf.transpose(eigvec), [self.bond_d**2, self.bond_d, self.bond_d])
        #
        L_AAbar = tf.einsum("ab,sac,tbd->stcd", dom_eigmat, self.A, self.A)
        L_AAbar_Rk = tf.einsum("stcd,kcd->kst", L_AAbar, eigtens)
        L_AAbar_Rk_Z = tf.einsum("kst,st->k", L_AAbar_Rk, operator)
        #
        AAbar_R = tf.einsum("sac,tbd,cd->stab", self.A, self.A, dom_eigmat)
        Lk_AAbar_R = tf.einsum("kab,stab->kst", eigtens, AAbar_R)
        Lk_AAbar_R_Z = tf.einsum("kst,st->k", Lk_AAbar_R, operator)
        #
        ss_list = []
        for n in np.arange(1,range):
            delta = (n-1) * tf.ones([self.bond_d ** 2], tf.float64)
            we = tf.reduce_sum(L_AAbar_Rk_Z * Lk_AAbar_R_Z * tf.pow(eigval, delta)) / dom_eigval ** (n + 1)
            ss_list.append(we)

        return ss_list

    def correlator_left_canonical_mps(self, operator, range):

        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        AAbar_R = tf.einsum("udf,veg,fg->uvde", self.A, self.A, right_eigenmatrix)
        #
        T = tf.einsum("sab,scd->acbd", self.A, self.A)
        i = tf.constant(0)
        iden = tf.einsum("bd,ce->bcde", tf.eye(self.bond_d,dtype=tf.float64),tf.eye(self.bond_d,dtype=tf.float64))
        #
        ss_list = []
        for n in np.arange(0, range):
            condition = lambda i, next: tf.less(i, n)
            body = lambda i, next: (tf.add(i, 1), tf.einsum("abcd,cdef->abef", T, next))
            i_fin, T_pow = tf.while_loop(condition, body, [i, iden])
            # No need for normalization in the LCF
            we = tf.einsum("stbc,st,bcde,uvde,uv->", AAbar, operator, T_pow, AAbar_R, operator)
            ss_list.append(we)

        return ss_list

    def single_site_expectation_value_left_canonical_mps(self, operator):

        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        exp_val = tf.einsum("sab,tac,bc,st->", self.A, self.A, right_eigenmatrix, operator)

        return exp_val


    @property
    def entanglement_spectrum(self):
        """
        Calculate the spectrum of eigenvalues of the reduced density matrix for a bipartition of
        an infinite system into two semi-infinite subsystems.

        :return: The `bond_d` eigenvalues of the reduced density matrix
        """
        pass

    # TODO Calculation of entanglement spectrum

    @property
    def transfer_matrix(self):
        if self._transfer_matrix is None:
            T = tf.einsum("sab,scd->acbd", self.A, self.A)
            T = tf.reshape(T, [self.bond_d**2, self.bond_d**2])
            self._transfer_matrix = T
        return self._transfer_matrix

    @property
    def right_eigenvector(self):
        if self._right_eigenvector is None:
            T = self.transfer_matrix
            vec = tf.ones([self.bond_d ** 2], dtype=tf.float64)
            next_vec = tf.einsum("ab,b->a", T, vec)
            # norm_big = lambda vec, next: tf.greater(tf.norm(vec - next), 1e-7)
            # CONDITION ON THE CHANGE OF VECTOR ELEMENTS, INSTEAD OF CHANGE OF THE NORM: r_prec
            norm_big = lambda vec, next: tf.reduce_all(
                tf.greater(tf.abs(vec - next), tf.constant(self.r_prec, shape=[self.bond_d ** 2], dtype=tf.float64)))
            increment = lambda vec, next: (next, tf.einsum("ab,b->a", T, next))
            vec, next_vec = tf.while_loop(norm_big, increment, [vec, next_vec])
            # Normalize using left vector
            left_vec = tf.reshape(tf.eye(self.bond_d, dtype=tf.float64), [self.bond_d ** 2])
            norm = tf.einsum('a,a->', left_vec, next_vec)
            self._right_eigenvector =  next_vec / norm
        return self._right_eigenvector


    @property
    def dominant_eig(self):
        if self._dominant_eig is None:
            eigvals, eigvecs = self._all_eig
            # We use cast to make the number an integer
            idx = tf.cast(tf.argmax(tf.abs(eigvals)), dtype=np.int32)# Why do abs?
            self._dominant_eig = eigvals[idx], eigvecs[:,idx] # Note that eigenvectors are given in columns, not rows!
        return self._dominant_eig

    def _symmetrize(self, M):
        # Symmetrize -- sufficient to guarantee transfer matrix is symmetric (but not necessary)
        M_lower = tf.matrix_band_part(M, -1, 0) #takes the lower triangular part of M (including the diagonal)
        return (M_lower + tf.matrix_transpose(M_lower)) / 2


    def _add_variational_energy_symmetric_mps(self, hamiltonian):
        """
        Evaluate the variational energy density for symmetric MPS (not using canonical form)

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        dom_eigval, dom_eigvec = self.dominant_eig
        dom_eigmat = tf.reshape(dom_eigvec, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("ab,sac,tbd->stcd", dom_eigmat, self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, dom_eigmat)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)

        return h_exp / tf.square(dom_eigval)

    def _add_variational_energy_left_canonical_mps(self, hamiltonian):
        """
        Evaluate the variational energy density for MPS in left canonical form

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, right_eigenmatrix)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)

        return h_exp

    def _add_variational_energy_left_canonical_mps_onsite_and_NN(self, h_NN, h_onsite):
        """
        Evaluate the variational energy density for MPS in left canonical form

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian: h_NN + h_onsite,
        e.g. Transverse Field Ising. Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("sab,tac->stbc", self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, right_eigenmatrix)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp_NN = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, h_NN)
        #
        right_eigenmatrix = tf.reshape(self.right_eigenvector, [self.bond_d, self.bond_d])
        h_exp_onsite = tf.einsum("sab,tac,bc,st->", self.A, self.A, right_eigenmatrix, h_onsite)

        return h_exp_NN + h_exp_onsite


if __name__ == "__main__":
    #################################ss
    # TRANSVERSE FIELD ISING
    #################################
    # physical and bond dimensions of MPS
    phys_d = 2
    bond_d = 4
    r_prec = 1e-14 #convergence condition for right eigenvector
    # Hamiltonian parameters
    J = 1
    h = 0.48

    # Pauli spin=1/2 matrices. For now we avoid complex numbers
    X = tf.constant([[0, 1], [1, 0]], dtype=tf.float64)
    iY = tf.constant([[0, 1], [-1, 0]], dtype=tf.float64)
    Z = tf.constant([[1, 0], [0, -1]], dtype=tf.float64)

    I = tf.eye(phys_d, dtype=tf.float64)

    XX = tf.einsum('ij,kl->ikjl', X, X)
    YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    ZZ = tf.einsum('ij,kl->ikjl', Z, Z)
    # X1 = (tf.einsum('ij,kl->ikjl', X, I) + tf.einsum('ij,kl->ikjl', I, X)) / 2
    X1 = tf.einsum('ij,kl->ikjl', X, I)

    # Heisenberg Hamiltonian
    # My impression is that staggered correlations go hand in hand with nonsymmetric A matrices
    h_xxx = XX + YY + ZZ

    h_zz = tf.constant(J / 4, dtype=tf.float64)
    h_x1 = tf.constant(h / 2, dtype=tf.float64)
    # Ising Hamiltonian (at criticality). Exact energy is -4/pi=-1.27324...
    h_ising = -h_zz * ZZ - h_x1 * X1
    # h_ising = - ZZ - X1



    #################################
    #AKLT
    #################################

    # phys_d = 3
    # bond_d = 2
    # r_prec = 1e-14

    # # Follow Annals of Physics Volume 326, Issue 1, Pages 96-192.
    # # Note that even though the As are not symmetric, the transfer matrix is.
    # # We normalize these to be in left (and right) canonical form
    #
    # Aplus = np.array([[0, 1 / np.sqrt(2)], [0, 0]])
    # Aminus = np.array([[0, 0], [-1 / np.sqrt(2), 0]])
    # A0 = np.array([[-1 / 2, 0], [0, 1 / 2]])
    # A_matrices = np.array([Aplus, A0, Aminus]) * np.sqrt(4 / 3)
    #
    # # Spin 1 operators.
    #
    # X = tf.constant([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
    # iY = tf.constant([[0, -1, 0], [1, 0, -1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
    # Z = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=tf.float64)
    #
    # XX = tf.einsum('ij,kl->ikjl', X, X)
    # YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    # ZZ = tf.einsum('ij,kl->ikjl', Z, Z)
    #
    # hberg = XX + YY + ZZ
    # h_aklt = hberg + tf.einsum('abcd,cdef->abef', hberg, hberg) / 3

    #######################################################################################
    #######################################################################################

    # Initialize the MPS


    imps = Tfimps(phys_d, bond_d, hamiltonian=h_ising, symmetrize=False)
    problem = pymanopt.Problem(manifold=imps.mps_manifold, cost=imps.variational_energy,
                               arg=imps.Stiefel)

    # learning_rate = 0.001

    with tf.Session() as sess:
        # point = sess.run(tf.reshape(imps.mps_manifold.rand(), [phys_d, bond_d, bond_d]))
        sess.run(tf.global_variables_initializer())
        # print(problem.cost(point))
        # solver = pymanopt.solvers.SteepestDescent(maxiter=5000,mingradnorm=1e-6)
        solver = pymanopt.solvers.ConjugateGradient(maxtime=float('inf'), maxiter=100000, mingradnorm=1e-20,
                                                    minstepsize=1e-20)
        Xopt = solver.solve(problem)
        print(Xopt)
        print(problem.cost(Xopt))

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(imps.A))
    #
    #     print(problem.cost(imps.A))
    # # solver = pymanopt.solvers.ConjugateGradient()

    # Xopt = solver.solve(problem)

    # imps = Tfimps(phys_d, bond_d, symmetrize=True, hamiltonian=h_ising)
    #
    # train_op = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(imps.variational_e)
    #
    # with tf.Session() as sess:
    #
    #     sess.run(tf.global_variables_initializer())
    #
    #     for i in range(100):
    #         print(sess.run([imps.variational_e, train_op])[0])