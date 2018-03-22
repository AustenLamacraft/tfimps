import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Optimization of infinite matrix product states in TensorFlow



tfe.enable_eager_execution()

# Assume spin-1/2 for now

bond_d = 5 # bond dimension of MPS

# Initialize MPS and add to graph
# TODO Need to symmetrize
A_init = np.random.rand(2, bond_d, bond_d)
A = tf.get_variable("A_matrices", initializer=A_init, trainable=True)

# The transfer matrix

T = tf.reduce_sum(A, axis=0)

# Pauli matrices. For now we avoid complex numbers

X = tf.constant([[0,1],[1,0]])
iY = tf.constant([[0,1],[-1,1]])
Z = tf.constant([[1,0],[0,-1]])

XX = tf.einsum('ij,kl->ikjl', X, X)
YY = - tf.einsum('ij,kl->ikjl', iY, iY)
ZZ = tf.einsum('ij,kl->ikjl', X, X)
print(XX, YY, ZZ)

# The Heisenberg Hamiltonian

H_XXX = XX + YY + ZZ


eigvals, eigvecs = tf.self_adjoint_eig(T)

print(eigvals)


# max_eigval = tf.reduce_max(tf.abs(eigvals))
#
# train_op = tf.train.GradientDescentOptimizer(max_eigval).minimize(-max_eigval)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(10):
#     print(sess.run([max_eigval, train_op])[0])