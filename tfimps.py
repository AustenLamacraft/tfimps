import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe

#tfe.enable_eager_execution()

# Optimization of infinite matrix product states in TensorFlow


class TFIMPS:

    def __init__(self, bond_d):
        self.bond_d = bond_d
        # Initialize MPS and add to computational graph
        # No need to symmetrize as only lower triangular part is used by eigensolver
        #  Assume spin-1/2 for now
        A_init = np.random.rand(2, bond_d, bond_d)
        self.A = tf.get_variable("A_matrices", initializer=A_init, trainable=True)

    def variational_e(self, hamiltonian):
        '''
        Evaluate the variational energy density.

        :param hamiltonian: tensor of shape [phys_d, phys_d, phys_d, phys_d] giving two-site Hamiltonian
        :return: expectation value of the energy density
        '''
        dom_eigval, dom_eigvec = self._dominant_eig
        return self.A / tf.sqrt(dom_eigval)

    @property
    def _transfer_matrix(self):
        T = tf.einsum("sab,scd->acbd", self.A, self.A)
        return tf.reshape(T, [self.bond_d**2, self.bond_d**2])

    @property
    def _dominant_eig(self):
        eigvals, eigvecs = tf.self_adjoint_eig(self._transfer_matrix)
        idx = tf.argmax(eigvals)
        return eigvals[idx], eigvecs[idx]




# bond dimension of MPS
bond_d = 5

imps = TFIMPS(bond_d)

# Pauli matrices. For now we avoid complex numbers

X = tf.constant([[0,1],[1,0]])
iY = tf.constant([[0,1],[-1,1]])
Z = tf.constant([[1,0],[0,-1]])

XX = tf.einsum('ij,kl->ikjl', X, X)
YY = - tf.einsum('ij,kl->ikjl', iY, iY)
ZZ = tf.einsum('ij,kl->ikjl', X, X)

# Heisenberg Hamiltonian
H_XXX = XX + YY + ZZ

with tf.Session() as sess:

    # Try normalizing
    sess.run(tf.global_variables_initializer())
    A_norm = imps.variational_e(H_XXX)
    print(sess.run(A_norm))



#
# train_op = tf.train.GradientDescentOptimizer(max_eigval).minimize(-max_eigval)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(10):
#     print(sess.run([max_eigval, train_op])[0])