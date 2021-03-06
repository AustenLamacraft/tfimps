import numpy as np
import tensorflow as tf
import tfimps
import pymanopt.manifolds
import pymanopt.solvers
import tensorflow as tf

class TestTfimps(tf.test.TestCase):

    def testMPSInLeftCanonicalForm(self):
        phys_d = 2
        bond_d = 3

        imps = tfimps.Tfimps(phys_d, bond_d, symmetrize=False)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            A = sess.run(imps.A)
            self.assertAllClose(np.tensordot(A, A, axes=([0, 1], [0, 1])), np.identity(bond_d))

    def testRightEigenvectorHasUnitEigenvalue(self):
        phys_d = 2
        bond_d = 4

        imps = tfimps.Tfimps(phys_d, bond_d, symmetrize=False)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            T = sess.run(imps.transfer_matrix)
            vec = sess.run(imps.right_eigenvector)
            self.assertAllClose(T@vec, vec)

    def testTransferMatrixForIdentity(self):
        phys_d = 2
        bond_d = 2

        A1 = A0 =  np.identity(phys_d)
        A_matrices = np.array([A0, A1])

        imps = tfimps.Tfimps(phys_d, bond_d, A_matrices)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(imps.transfer_matrix)
            self.assertAllClose(phys_d * np.identity(4), actual)

    def testDominantEigenvectorIsEigenvector(self):
        phys_d = 3
        bond_d = 5
        imps = tfimps.Tfimps(phys_d, bond_d)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            T = sess.run(imps.transfer_matrix)
            val, vec = sess.run(imps.dominant_eig)
            self.assertAllClose(T@vec, val*vec)

    def testIdentityHamiltonianHasEnergyOneDiagonalMPS(self):
        phys_d = 2
        bond_d = 5

        A0 = np.diag(np.random.rand(bond_d))
        A1 = np.diag(np.random.rand(bond_d))
        A_matrices = np.array([A0, A1])


        I = tf.eye(phys_d, dtype=tf.float64)
        h = tf.einsum('ij,kl->ikjl', I, I)

        imps = tfimps.Tfimps(phys_d, bond_d, A_matrices, hamiltonian=h)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(imps.variational_energy)
            self.assertAllClose(1, actual)

    def testIdentityHamiltonianHasEnergyOneRandomMPS(self):
        phys_d = 3
        bond_d = 5

        I = tf.eye(phys_d, dtype=tf.float64)
        h = tf.einsum('ij,kl->ikjl', I, I)

        imps = tfimps.Tfimps(phys_d, bond_d, hamiltonian=h)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(imps.variational_energy)
            self.assertAllClose(1, actual)

    def testAKLTStateHasCorrectEnergy(self):
        phys_d = 3
        bond_d = 2

        # Follow Annals of Physics Volume 326, Issue 1, Pages 96-192.
        # Note that even though the As are not symmetric, the transfer matrix is.
        # We normalize these to be in left (and right) canonical form

        Aplus = np.array([[0, 1/np.sqrt(2)], [0, 0]])
        Aminus = np.array([[0, 0], [-1/np.sqrt(2), 0]])
        A0 = np.array([[-1/2, 0], [0, 1/2]])
        A_matrices = np.array([Aplus, A0, Aminus]) * np.sqrt(4/3)

        # Spin 1 operators.

        X = tf.constant([[0, 1, 0 ], [1, 0, 1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        iY = tf.constant([[0, -1, 0 ], [1, 0, -1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        Z = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=tf.float64)

        XX = tf.einsum('ij,kl->ikjl', X, X)
        YY = - tf.einsum('ij,kl->ikjl', iY, iY)
        ZZ = tf.einsum('ij,kl->ikjl', Z, Z)

        hberg = XX + YY + ZZ
        h_aklt = hberg + tf.einsum('abcd,cdef->abef', hberg, hberg) / 3

        aklt = tfimps.Tfimps(phys_d, bond_d, A_matrices, symmetrize=False, hamiltonian=h_aklt)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            aklt_energy = sess.run(aklt.variational_energy)
            self.assertAllClose(-2/3, aklt_energy)

    def testAKLTStateHasCorrectCorrelations(self):
        phys_d = 3
        bond_d = 2


        # Follow Annals of Physics Volume 326, Issue 1, Pages 96-192.
        # AKLT correlations appear between Eqs. (115) and (116).
        # The tensors below correspond to not normalized state in the thermodynamic limit.
        # They should all be multiplied by sqrt(4/3) to get a normalized state.
        # One can also normalize the final result with the dominant eigenvalue.

        Aplus = np.array([[0, 1/np.sqrt(2)], [0, 0]])
        Aminus = np.array([[0, 0], [-1/np.sqrt(2), 0]])
        A0 = np.array([[-1/2, 0], [0, 1/2]])
        A_matrices = np.array([Aplus, A0, Aminus])

        aklt = tfimps.Tfimps(phys_d, bond_d, A_matrices, symmetrize=False)

        # Spin 1 operators.

        X = tf.constant([[0, 1, 0 ], [1, 0, 1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        iY = tf.constant([[0, -1, 0 ], [1, 0, -1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        Z = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=tf.float64)

        # Range of of values j-i

        range = 6

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            xx_eval = sess.run(aklt.correlator(Z, range))
            xx_exact = 12 / 9 * (-1/3)**np.arange(1,range)
            self.assertAllClose(xx_eval, xx_exact)

    def testAKLTStateHasCorrectCorrelationswithoutspectrum(self):
        phys_d = 3
        bond_d = 2


        # Follow Annals of Physics Volume 326, Issue 1, Pages 96-192.
        # AKLT correlations appear between Eqs. (115) and (116).
        # The tensors below correspond to not normalized state in the thermodynamic limit.
        # They should all be multiplied by sqrt(4/3) to get a normalized state.
        # One can also normalize the final result with the dominant eigenvalue.

        # Aplus = np.array([[0, 1/np.sqrt(2)], [0, 0]])
        # Aminus = np.array([[0, 0], [-1/np.sqrt(2), 0]])
        # A0 = np.array([[-1/2, 0], [0, 1/2]])
        Aplus = np.array([[0, np.sqrt(2 / 3)], [0, 0]])
        Aminus = np.array([[0, 0], [-np.sqrt(2 / 3), 0]])
        A0 = np.array([[-1 / np.sqrt(3), 0], [0, 1 / np.sqrt(3)]])
        A_matrices = np.array([Aplus, A0, Aminus])

        aklt = tfimps.Tfimps(phys_d, bond_d, A_matrices, symmetrize=False)

        # Spin 1 operators.

        X = tf.constant([[0, 1, 0 ], [1, 0, 1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        iY = tf.constant([[0, -1, 0 ], [1, 0, -1], [0, 1, 0]], dtype=tf.float64) / np.sqrt(2)
        Z = tf.constant([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=tf.float64)

        # Range of of values j-i

        range = 6

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            xx_eval = sess.run(aklt.correlator_left_canonical_mps(Z, range))
            xx_exact = 12 / 9 * (-1/3)**np.arange(1,range+1)
            self.assertAllClose(xx_eval, xx_exact)