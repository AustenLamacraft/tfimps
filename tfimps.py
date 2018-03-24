import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

# Optimization of infinite matrix product states in TensorFlow

class Tfimps:

    def __init__(self, phys_d, bond_d, bond_matrices=None):
        self.phys_d = phys_d
        self.bond_d = bond_d
        # Initialize MPS and add to computational graph
        # Only lower triangular part is used by eigensolver
        # Do we need to symmetrize for evaluation?
        if bond_matrices is None:
            A_init = np.random.rand(phys_d, bond_d, bond_d)

        else:
            A_init = bond_matrices

        # Symmetrize
        A_upper = tf.matrix_band_part(A_init, 0, -1)
        A_symm = 0.5 * (A_upper + tf.transpose(A_upper, [0,2,1]))
        self.A = tf.get_variable("A_matrices", initializer=A_symm, trainable=True)

    def variational_e(self, hamiltonian):
        """
        Evaluate the variational energy density.

        :param hamiltonian: tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian.
            Adopt convention that first two indices are row, second two are column.
        :return: expectation value of the energy density
        """
        dom_eigval, dom_eigvec = self._dominant_eig
        dom_eigmat = tf.reshape(dom_eigvec, [self.bond_d, self.bond_d])
        L_AAbar = tf.einsum("ab,sac,tbd->stcd", dom_eigmat, self.A, self.A)
        AAbar_R = tf.einsum("uac,vbd,cd->uvab", self.A, self.A, dom_eigmat)
        L_AAbar_AAbar_R = tf.einsum("stab,uvab->sutv", L_AAbar, AAbar_R)
        h_exp = tf.einsum("stuv,stuv->", L_AAbar_AAbar_R, hamiltonian)
        return h_exp / tf.square(dom_eigval)

    @property
    def _transfer_matrix(self):
        T = tf.einsum("sab,scd->acbd", self.A, self.A)
        T = tf.reshape(T, [self.bond_d**2, self.bond_d**2])
        return T

    @property
    def _dominant_eig(self):
        eigvals, eigvecs = tf.self_adjoint_eig(self._transfer_matrix)
        idx = tf.argmax(eigvals)
        return eigvals[idx], eigvecs[:,idx]


if __name__ == "__main__":

    # physical and bond dimensions of MPS
    phys_d = 2
    bond_d = 10

    imps = Tfimps(phys_d, bond_d)

    # Pauli matrices. For now we avoid complex numbers
    X = tf.constant([[0,1],[1,0]], dtype=tf.float64)
    iY = tf.constant([[0,1],[-1,1]], dtype=tf.float64)
    Z = tf.constant([[1,0],[0,-1]], dtype=tf.float64)

    XX = tf.einsum('ij,kl->ikjl', X, X)
    YY = - tf.einsum('ij,kl->ikjl', iY, iY)
    ZZ = tf.einsum('ij,kl->ikjl', Z, Z)

    # Heisenberg Hamiltonian
    H_XXX = XX + YY + ZZ

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print(sess.run(imps.variational_e(H_XXX)))




#
# train_op = tf.train.GradientDescentOptimizer(max_eigval).minimize(-max_eigval)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(10):
#     print(sess.run([max_eigval, train_op])[0])