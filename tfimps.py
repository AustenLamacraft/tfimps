import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

class Tfimps:
    """
    Infinite Matrix Product State class.
    """

    # TODO Allow for two-site unit cell and average energy between A_1 A_2 and A_2 A_1 ordering
    def __init__(self, phys_d, bond_d, A_matrices=None, symmetrize=True):
        """
        :param phys_d: Physical dimension of the state e.g. 2 for spin-1/2 systems.
        :param bond_d: Bond dimension, the size of the A matrices.
        :param A_matrices: Square matrices of size `bond_d` forming the Matrix Product State.
        :param symmetrize: Boolean indicating A matrices are symmetrized.
        """
        self.phys_d = phys_d
        self.bond_d = bond_d

        if A_matrices is None:
            A_init = self._symmetrize(np.random.rand(phys_d, bond_d, bond_d))

        else:
            A_init = A_matrices

        self.A = tf.get_variable("A_matrices", initializer=A_init, trainable=True)

        if symmetrize:
            self.A = self._symmetrize(self.A)

    def variational_e(self, hamiltonian):
        """
        Evaluate the variational energy density.

        :param hamiltonian: Tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: Expectation value of the energy density.
        """
        dom_eigval, dom_eigvec = self._dominant_eig
        dom_eigmat = tf.reshape(dom_eigvec, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("ab,sac,tbd->stcd", dom_eigmat, self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, dom_eigmat)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)
        return h_exp / tf.square(dom_eigval)

    # TODO Method for correlation functions
    def correlator(self, operator, range):
        """
        Evaluate the correlation function of `operator` up to `range` sites.

        :param operator: Tensor of shape [phys_d, phys_d] giving single site operator.
        :param range: Maximum separation at which correlations required
        :return: Correlation function
        """
        pass

    # TODO Calculation of entanglement spectrum
    @property
    def entanglement_spectrum(self):
        """
        Calculate the spectrum of eigenvalues of the reduced density matrix for a bipartition of
        an infinite system into two semi-infinite subsystems.

        :return: The `bond_d` eigenvalues of the reduced density matrix
        """
        pass

    @property
    def _transfer_matrix(self):
        T = tf.einsum("sab,scd->acbd", self.A, self.A)
        T = tf.reshape(T, [self.bond_d**2, self.bond_d**2])
        return T

    @property
    def _dominant_eig(self):
        eigvals, eigvecs = tf.self_adjoint_eig(self._transfer_matrix)
        idx = tf.cast(tf.argmax(tf.abs(eigvals)), dtype=np.int32)
        return eigvals[idx], eigvecs[:,idx]

    def _symmetrize(self, M):
        # Symmetrize -- sufficient to guarantee transfer matrix is symmetric (but not necessary)
        M_lower = tf.matrix_band_part(M, -1, 0)
        return (M_lower + tf.matrix_transpose(M_lower)) / 2

if __name__ == "__main__":

    # physical and bond dimensions of MPS
    phys_d = 2
    bond_d = 16

    imps = Tfimps(phys_d, bond_d, symmetrize=True)

    # Pauli matrices. For now we avoid complex numbers
    X = tf.constant([[0,1],[1,0]], dtype=tf.float64)
    iY = tf.constant([[0,1],[-1,0]], dtype=tf.float64)
    Z = tf.constant([[1,0],[0,-1]], dtype=tf.float64)

    I = tf.eye(phys_d, dtype=tf.float64)

    XX = tf.einsum('ij,kl->ikjl', X, X)
    YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    ZZ = tf.einsum('ij,kl->ikjl', Z, Z)
    X1 = (tf.einsum('ij,kl->ikjl', X, I) + tf.einsum('ij,kl->ikjl', I, X)) / 2


    # Heisenberg Hamiltonian
    # My impression is that staggered correlations go hand in hand with nonsymmetric A matrices
    h_xxx = XX + YY + ZZ

    # Ising Hamiltonian (at criticality). Exact energy is -4/pi=-1.27324...
    h_ising = - ZZ - X1

    train_op = tf.train.AdamOptimizer(learning_rate = 0.005).minimize(imps.variational_e(h_ising))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(200):
            print(sess.run([imps.variational_e(h_ising), train_op])[0])