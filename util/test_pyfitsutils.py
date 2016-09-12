#/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
#*/
from __future__ import print_function
from __future__ import absolute_import

from .fits import *
import numpy as np
import unittest


class TestSlices(unittest.TestCase):
    def setUp(self):
        T = tabledata()
        T.x1 = np.arange(10).astype(int)
        T.x2 = np.arange(10).astype(float)
        T.x3 = [i for i in range(10)]
        T.x4 = tuple([i for i in range(10)])
        self.assertEqual(len(T), 10)
        T.about()
        self.T = T

    def testCut(self):
        T1 = self.T.copy()
        T1.about()

        T1.cut(np.arange(10) < 4)
        T1.about()
        self.assertEqual(len(T1), 4)

    def testIter(self):
        for i in range(len(self.T)):
            self.T[i].about()
        for t in self.T:
            t.about()

    def testNumpySlicing(self):
        A = np.arange(10)
        print(A[:3])
        print(A[3:6])
        print(A[7:])
        print(A[1:7:2])

        A = list(A)
        print(A[:3])
        print(A[3:6])
        print(A[7:])
        print(A[1:7:2])

        A = np.arange(10)
        A = A[:,np.newaxis].repeat(2, axis=1)
        print(A[:3])
        print(A[3:6])
        print(A[7:])
        print(A[1:7:2])

    def testSlice2(self):
        T1 = self.T[self.T.x1 < 4]
        self.assertEqual(len(T1), 4)
        T1.about()
        self.assertTrue(all(T1.x1 == np.array([0,1,2,3])))
        self.assertEqual(T1.x3, [0,1,2,3])

        T1 = self.T[np.flatnonzero(self.T.x1 < 4)]
        self.assertEqual(len(T1), 4)
        T1.about()
        self.assertTrue(all(T1.x1 == np.array([0,1,2,3])))
        self.assertEqual(T1.x3, [0,1,2,3])

    def testSlice1(self):
        T1 = self.T[:3]
        self.assertEqual(len(T1), 3)
        T1.about()
        self.assertTrue(all(T1.x1 == np.arange(3)))
        self.assertTrue(all(T1.x2 == np.arange(3).astype(float)))
        self.assertEqual(T1.x3, [0,1,2])
        self.assertEqual(T1.x4, (0,1,2))

        T2 = self.T[3:6]
        self.assertEqual(len(T2), 3)
        T2.about()
        self.assertEqual(T2.x3, [3,4,5])
        self.assertEqual(T2.x4, (3,4,5))

        T3 = self.T[7:]
        self.assertEqual(len(T3), 3)
        T3.about()
        self.assertEqual(T3.x3, [7,8,9])

        T4 = self.T[1:7:2]
        self.assertEqual(len(T4), 3)
        T4.about()
        self.assertEqual(T4.x3, [1,3,5])


if __name__ == '__main__':
    unittest.main()



