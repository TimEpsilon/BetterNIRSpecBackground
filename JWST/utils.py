from scipy.interpolate import CloughTocher2DInterpolator as CT
import numpy as np
import json
import logging

logger = logging.getLogger("stpipe")


def logConsole(text, source=None):
	"""
	 Logger : displays time + log

	 Source can be WARNING, ERROR, DEBUG or None / INFO / any other string,
	 in which case it will be considered an INFO log
	"""
	text = f" - [BetterBackground]  : {text}"
	logType = {"WARNING": lambda : logger.warning(text),
			   "ERROR": lambda : logger.error(text),
			   "DEBUG": lambda : logger.debug(text)
			   }

	logType.get(source,lambda : logger.info(text))()



def WhichShutterOpen(shutter_state):
	"""
	 Gets the id of which shutter is open in the slitlet.
	 If SHUTSTA is unusual (i.e. only one shutter is open), returns None
	"""
	if shutter_state == "11x":
		return 2
	elif shutter_state == "1x1":
		return 1
	elif shutter_state == "x11":
		return 0
	else:
		return None



def gaussian(x,x0,A,s):
	"""
	Simple gaussian function
	"""
	return A * np.exp(-(x-x0)**2 / (2*s**2))

"""
Slitlet model : 3 gaussians of same sigma
"""
def slitletModel(x,x1,x2,x3,A1,A2,A3,s,c):
	return gaussian(x,x1,A1,s) + gaussian(x,x2,A2,s) + gaussian(x,x3,A3,s) + c


def IDWExtrapolation(xy, ui, power=2):
	"""
	Rough implementation of the Inverse Distance Weighting algorithm

	Parameters
	----------
	xy : ndarray, shape (npoints, ndim)
		Coordinates of data points
	ui : ndarray, shape (npoints)
		Values at data points

	Returns
	-------
	func : callable
	"""
	x = xy[:, 0]
	y = xy[:, 1]

	def new_f(xx,yy):

		xy_ravel = np.column_stack((xx.ravel(),yy.ravel()))
		x_ravel = xy_ravel[:, 0]
		y_ravel = xy_ravel[:, 1]

		X1, X2 = np.meshgrid(x,x_ravel)
		Y1, Y2 = np.meshgrid(y,y_ravel)

		d = ((X1-X2)**2 + (Y1-Y2)**2).T

		w = np.zeros_like(d,dtype=float)

		w[d>0] = d[d>0]**(-power/2)

		w_ui_sum = ui[:, None]*w
		w_ui_sum = w_ui_sum.sum(axis=0)

		wsum = w.sum(axis=0)

		result = w_ui_sum / wsum
		result = result.reshape(np.shape(xx))

		result[x,y] = ui
		return result

	return new_f


def NNExtrapolation(xy, z):
	"""
	Code From https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html

	CT interpolator + nearest-neighbor extrapolation.

	Parameters
	----------
	xy : ndarray, shape (npoints, ndim)
		Coordinates of data points
	z : ndarray, shape (npoints)
		Values at data points

	Returns
	-------
	func : callable
		A callable object which mirrors the CT behavior,
		with an additional neareast-neighbor extrapolation
		outside of the data range.
	"""
	x = xy[:, 0]
	y = xy[:, 1]
	f = CT(xy, z)

	# this inner function will be returned to a user
	def new_f(xx, yy):
		# evaluate the CT interpolator. Out-of-bounds values are nan.
		zz = f(xx, yy)
		nans = np.isnan(zz)

		if nans.any():
			# for each nan point, find its nearest neighbor
			inds = np.argmin(
				(x[:, None] - xx[nans])**2 +
				(y[:, None] - yy[nans])**2
				, axis=0)
			# ... and use its value
			zz[nans] = z[inds]
		return zz

	return new_f


def polynomialBasis(x, y, max_order=4):
	"""Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
	basis = []
	for i in range(max_order+1):
		for j in range(max_order - i +1):
			basis.append(x**j * y**i)
	return basis

def rewriteJSON(file):
	"""
	Rewrites the asn.json files in order to apply to the _bkg files

	Parameters
	----------
	file : str
	Path to the asn.json file

	Returns
	-------
	"""

	with open(file, "r") as asn:
		data = json.load(asn)

		# Get calibration indices
		not_science = []
		for i in range(len(data["products"][0]["members"])):
			data["products"][0]["members"][i]["expname"] = data["products"][0]["members"][i]["expname"].replace("_cal",
			"_bkg_photomstep")
			if not data["products"][0]["members"][i]["exptype"] == "science":
				not_science.append(i)

		# Starting from the end in order to keep the values at the same index
		for i in not_science[::-1]:
			del data["products"][0]["members"][i]

	with open(file, "w") as asn:
		json.dump(data, asn, indent=4)

def numberSameLength(entry):
	"""
	Prepends 0 to a number in order to respect the XXXXX format
	----------
	entry : a number, assumed to be < 5 chars long

	Returns
		a str with 0 prepended to a number
	-------

	"""
	entry = [*str(entry)]
	while len(entry) < 5:
		entry.insert(0, '0')
	entry = "".join(entry)
	return entry

def getPolynomialOrder(coeffCount):
	"""
	Returns the order of a polynomial based on the amount of coefficients

	The amount of coefficients is given by n^2/2 + 3/2*n + 1, n the order
	Parameters

	The order is assumed to be somewhere between 0 and 6
	----------
	coeffCount : int
		The amount of coefficients

	Returns
	-------
	order : int
		The order of the polynomial
	"""

	order = None
	for n in range(7):
		amount = n**2 / 2 + 3/2 * n + 1
		if amount == coeffCount:
			order = n
			return order

	return order

def getCRDSPath():
	with open("../CRDS_PATH.txt") as file:
		txt = file.readline()
		if txt is None or txt == "":
			raise Exception("CRDS_PATH.txt not found or file empty")
		return txt