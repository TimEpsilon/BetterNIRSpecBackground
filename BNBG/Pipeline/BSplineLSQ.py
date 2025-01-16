import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

class BSplineLSQ:
	def __init__(self, x : np.ndarray, y : np.ndarray, w : np.ndarray, t=None, n=0.1, k=4, curvatureConstraint=1, endpointConstraint=0.1):
		"""
		A simple class for BSpline objects. Relies on the BSplines objects of scipy,
		but allows for more control on the fitting than the usual LSQ algorithm of scipy.interpolate.

		Parameters
		----------
		x : ndarray
			1D array of x values

		y : ndarray
			1D array of corresponding y values

		w : ndarray
			1D array of weights, usually 1/dy. NaNs will break the code so they need to be filtered beforehand

		t : ndarray
			The knots of the spline. Is of the shape [(k-1),(n-k+2),(k-1)], the n-k+2 points being interior knots

		n : float
			In range [0, 1], the fraction of len(x) used to determine the amount of interior knots

		k : int
			The rank of the spline, i.e. the polynomials will be of order k-1

		curvatureConstraint : float
			An hyperparameter used for regularizing, i.e., how much the curvature will be minimized. Useful if gaps are present in the data
			If equal to 0, this will entirely ignore curvature

		endpointConstraint : float
			An hyperparameter used for the endpoints, i.e., how much the slope on each side of the spline will be minimized.
			If equal to 0, this will entirely ignore the endpoint slopes
		"""

		self.x = x
		self.y = y
		self.w = w
		self.k = k
		self.nInsideKnots = max(int(n * len(x)), k - 1)
		self.nAllKnots = self.nInsideKnots + 2 * (k - 1)
		self.ncoeffs = self.nAllKnots - k - 1
		self.curvatureConstraint = curvatureConstraint
		self.endpointConstraint = endpointConstraint

		# Uniform knots based on the data distribution
		# i.e. if there is a higher density of points, there will be more knots
		if t is None:
			self.t = self._regularSpacedKnots()
		else:
			self.t = t

		self.X, A1_a, A1_b, A2 = self._getSplineMatrices()

		# Creating the matrix A1.T @ A1
		self.A1_a = np.outer(A1_a, A1_a)
		self.A1_b = np.outer(A1_b, A1_b)

		# Weight matrix
		self.W = np.diagflat(w)

		self.G2 = self._getGramMatrix(A2)

		self.invcov, self.cov, self.c, self.spline = self._calculateCoefficients()

	def __call__(self, x):
		return self.spline(x)

	def getError(self, x : np.ndarray):
		"""
		Calculates the error of the spline df = sqrt(B(x).T @ cov @ B(x))
		Parameters
		----------
		x : ndarray
			Values at which to evaluate the error. Should work for arrays of any shape

		Returns
		-------
		Error at x on spl(x)

		"""
		shape = x.shape
		x = x.ravel()

		B = np.zeros((len(x), self.ncoeffs))
		for i in range(self.ncoeffs):
			c = np.zeros(self.ncoeffs)
			c[i] = 1
			spl = BSpline(self.t, c, self.k)

			B[:, i] = spl(x)

		df = np.zeros((len(x)))
		for j in range(len(x)):
			b = B[j,:]
			df[j] = np.sqrt(b.T @ self.cov @ b)
		return df.reshape(shape)

	def plot(self, ax):
		"""
		Plots the spline and the data, along with 2 sliders allowing to calculate on the fly the effect of both hyperparameters.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			Axis on which to plot the data

		Returns
		-------

		"""

		# Slider
		plt.subplots_adjust(bottom=0.2)
		ax_slider1 = plt.axes((0.2, 0.1, 0.6, 0.03))
		slider1 = Slider(ax_slider1, 'CurvatureConstraint', 0, 5, valinit=self.curvatureConstraint, valstep=0.01)
		ax_slider2 = plt.axes((0.2, 0.05, 0.6, 0.03))
		slider2 = Slider(ax_slider2, 'EndpointConstraint', 0, 1, valinit=self.endpointConstraint, valstep=0.01)

		def update(val):
			ax[0].clear()
			ax[1].clear()

			c1 = slider1.val
			c2 = slider2.val

			self.updateParameters(c1, c2)

			ax[0].grid()

			ax[0].scatter(self.x, self.y, color='k', alpha=0.1, marker='+')
			ax[0].plot(self.x, 1 / self.w, color='r', linewidth=1)
			x = np.linspace(np.min(self.x), np.max(self.x), 300)
			y = self(x)
			dy = self.getError(x)
			ax[0].plot(x, y, color='b')
			ylim = ax[0].get_ylim()
			ax[0].fill_between(x, y - dy, y + dy, color='b', alpha=0.1)
			ax[0].scatter(self.t, self(self.t), color='b', marker='D')
			ax[0].set_ylim(*ylim)

			ax[1].imshow(np.abs(self.W), norm=LogNorm(), cmap='plasma')

		update(0)

		# Attach the update function to the slider
		slider1.on_changed(update)
		slider2.on_changed(update)

		return slider1, slider2

	def updateParameters(self, lambdaRegularization=None, lambdaEndpoints=None):
		"""
		Recalculates the spline with new values for both hyperparameters.

		Parameters
		----------
		lambdaRegularization : float
			New value for the curvature constraint

		lambdaEndpoints : float
			New value for the endpoints constraint

		"""
		if lambdaRegularization is not None:
			self.curvatureConstraint = lambdaRegularization
		if lambdaEndpoints is not None:
			self.endpointConstraint = lambdaEndpoints
		self.invcov, self.cov, self.c, self.spline = self._calculateCoefficients()

	########################
	# Private
	########################

	def _regularSpacedKnots(self):
		cdf = np.linspace(0, 1, len(self.x))
		targetCdf = np.linspace(0, 1, self.nInsideKnots)
		t = np.interp(targetCdf, cdf, self.x.copy())

		t0 = np.repeat(t[0], self.k - 1)
		tn = np.repeat(t[-1], self.k - 1)
		t = np.append(t0, t)
		t = np.append(t, tn)
		return t

	def _getSplineMatrices(self):
		# Initialise various matrices
		A1_a = np.zeros(self.ncoeffs)
		A1_b = np.zeros(self.ncoeffs)
		X = np.zeros((len(self.x), self.ncoeffs))
		A2 = np.zeros((self.ncoeffs, len(self.x)))

		# Iterate over every BSpline in the basis
		for i in range(self.ncoeffs):
			# Get i-th BSpline
			c = np.zeros(self.ncoeffs)
			c[i] = 1
			spl = BSpline(self.t, c, self.k)

			# X[j,i] = BSpline[i](x[j]), basically sampling each Bspline vector at each data point
			X[:, i] = spl(self.x)

			# Building 2 arrays (1 per endpoint) which measure the effect of each BSpline on the curvature
			_ = spl.derivative(1)(np.array([self.x[0], self.x[-1]]))
			A1_a[i] = _[0]
			A1_b[i] = _[1]

			# A[i,j] = BSpline[i]''(x[j]), sampling each Bspline vector curvature at each data point
			_ = spl.derivative(2)(self.x)
			A2[i, :] = _

		return X, A1_a, A1_b, A2

	def _getGramMatrix(self, A2):
		# Building a 3D array where 2 first axis are for each BSpline and the 3rd axis is for x values
		# G is constructed by integrating along the 3rd axis
		A22 = np.zeros((self.ncoeffs, self.ncoeffs, len(self.x)))
		for i in range(len(self.x)):
			# For a given x, calculate A2(x).T @ A2(x)
			A22[:, :, i] = np.outer(A2[:, i], A2[:, i])
		return np.trapezoid(A22, x=self.x, axis=2)

	def _calculateCoefficients(self):
		invcov = ((self.X.T @ self.W @ self.X)  # Basic LSQ normal form
				  + self.curvatureConstraint ** 2 * self.G2  # Curvature correction
				  + self.endpointConstraint ** 2 * (self.A1_b + self.A1_a))  # Endpoint correction
		cov = np.linalg.inv(invcov)

		# Final Coefficients
		c = cov @ self.X.T @ self.W @ self.y
		spline = BSpline(self.t, c, self.k)
		return invcov, cov, c, spline