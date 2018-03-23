import numpy as np
import tensorflow as tf
import tfimps

class TestTfimps(tf.test.TestCase):

    def testIdentityHamiltonianHasEnergyOne(self):
        phys_d = 3
        bond_d = 5
        imps = tfimps.Tfimps(phys_d, bond_d)
        I = tf.eye(phys_d, dtype=tf.float64)
        h = tf.einsum('ij,kl->ikjl', I, I)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertAllClose(1, imps.variational_e(h))

    def testAKLTEnergy(self):
        self.fail()
