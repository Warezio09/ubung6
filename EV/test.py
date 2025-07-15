import unittest
import numpy as np
from ev import power_method

class TestEV(unittest.TestCase):
    def test_eigenpair(self):
        A = np.array([[2, 0], [0, 3]])
        v0 = np.array([1, 1])
        l, x = power_method(A, v0)
        with self.subTest("Test for correct eigenvalue"):
            self.assertAlmostEqual(l[-1],3)
        with self.subTest("Test for correct eigenvector"):
            self.assertTrue(np.allclose(x, np.array([0, 1]), atol=1e-6))



if __name__ == '__main__':
    unittest.main()