import numpy as np
import tensorflow as tf
#import tensorflow.contrib.eager as tfe

#tfe.enable_eager_execution()

# Optimization of infinite matrix product states in TensorFlow

def transfer_matrix(A):
    bond_d = A.get_shape().as_list()[1]
    T = tf.einsum("sab,scd->acbd", A, A)
    return tf.reshape(T, [bond_d**2, bond_d**2])

def dominant_eig(T):
    eigvals, eigvecs = tf.self_adjoint_eig(T)
    idx = tf.argmax(eigvals)
    return eigvals[idx], eigvecs[idx]


# bond dimension of MPS
bond_d = 5

# Initialize MPS and add to computational graph
# No need to symmetrize as only lower triangular part is used by eigensolver
# Assume spin-1/2 for now
A_init = np.random.rand(2, bond_d, bond_d)
A = tf.get_variable("A_matrices", initializer=A_init, trainable=True)

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

    # Confirm eigenvectors are working
    sess.run(tf.global_variables_initializer())
    print(sess.run(dominant_eig(transfer_matrix(A))))



#
# train_op = tf.train.GradientDescentOptimizer(max_eigval).minimize(-max_eigval)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(10):
#     print(sess.run([max_eigval, train_op])[0])