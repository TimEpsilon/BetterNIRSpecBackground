import sys
sys.path.append("../MathFunctions")

import unittest
import numpy as np
from DataSimulating import MathFunctions

class DataTest_UT(unittest.TestCase):
	def testAdd2Dto1D(self):
		array2D = np.array([[1,2],
							[3,4],
							[5,6]])

		array1D = np.array([1])
		expect = np.array([[[2],[3]],
						   [[4],[5]],
						   [[6],[7]]])
		result = MathFunctions.add2Dto1D(array2D,array1D)
		self.assertEqual( True, np.all(expect==result), msg=f"Expected {expect} but got {result}")


		array1D = [1]
		result = MathFunctions.add2Dto1D(array2D, array1D)
		self.assertEqual(True, np.all(expect == result), msg=f"Expected {expect} but got {result}")


		array1D = 1
		result = MathFunctions.add2Dto1D(array2D, array1D)
		self.assertEqual(True, np.all(expect == result), msg=f"Expected {expect} but got {result}")


		array1D = np.array([1,2])
		expect = np.array([[[2,3],[3,4]],
						   [[4,5],[5,6]],
						   [[6,7],[7,8]]])
		result = MathFunctions.add2Dto1D(array2D, array1D)
		self.assertEqual( True, np.all(expect==result), msg=f"Expected {expect} but got {result}")


		array1D = np.array([1, 2, 3])
		expect = np.array([[[2, 3, 4], [3, 4, 5]],
						   [[4, 5, 6], [5, 6, 7]],
						   [[6, 7, 8], [7, 8, 9]]])
		result = MathFunctions.add2Dto1D(array2D, array1D)
		self.assertEqual( True, np.all(expect==result), msg=f"Expected {expect} but got {result}")


		print("Test for add2Dto1D passed")


	def testGauss(self):
		x = 3
		y = MathFunctions.gauss(x,3,1)
		self.assertEqual(1,y)

		x = 4
		y = MathFunctions.gauss(x,3,1)
		self.assertEqual(np.e**(-1/2),y)

		print("Test for gauss passed")


	def testLorentzian(self):
		x = 3
		y = MathFunctions.lorentzian(x,3,0.5)
		self.assertEqual(np.array([[2/(0.5*np.pi)]]),y)

		x = 3.25
		y = MathFunctions.lorentzian(x, 3, 0.5)
		self.assertEqual(np.array([[1 / (0.5 * np.pi)]]), y)

		print("Test for lorentzian passed")

	def testCast1Dto3D(self):
		array1D = np.array([1,2])
		shape = (2,2,2)
		expect = np.array([[[1,2],
							  [1,2]],
							 [[1,2],
							  [1,2]]])
		result = MathFunctions.cast1Dto3D(array1D,shape)
		self.assertEqual(True, np.all(expect==result), msg=f"Expected {expect} but got {result}")


		shape = (3,2,2)
		expect = np.array([[[1,2],
							[1,2]],
						   [[1,2],
							[1,2]],
						   [[1,2],
							[1,2]]])
		result = MathFunctions.cast1Dto3D(array1D, shape)
		self.assertEqual(True, np.all(expect == result), msg=f"Expected {expect} but got {result}")

		shape = (3,2,1)
		self.assertRaises(ValueError, lambda : MathFunctions.cast1Dto3D(array1D, shape))

		print("Test for cast1Dto3D passed")

if __name__ == '__main__':
	unittest.main()
