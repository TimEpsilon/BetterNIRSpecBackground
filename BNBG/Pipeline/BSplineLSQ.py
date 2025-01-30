import numpy as np

class BSplineLSQ:
	def __init__(self, x : np.ndarray, y : np.ndarray):
		"""
		Rewritten for testing. Since x and y are obtained from a median spectral sampling of the 2d image,
		this is equivalent to an exact fitting of the data (interpolationKnots = 1).
		The np.interp simply fills the gaps linearly

		Parameters
		----------
		x : ndarray
			1D array of x values

		y : ndarray
			1D array of corresponding y values
		"""
		self.x = x
		self.y = y
		self.spline = lambda X : np.interp(X, self.x, self.y)

	def __call__(self, x):
		return self.spline(x)