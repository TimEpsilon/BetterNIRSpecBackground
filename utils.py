import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator as CT
import numpy as np
import datetime
import time
from scipy.optimize import curve_fit as cfit
import json

"""
 Logger : displays time + log
"""
def logConsole(text, source=None):
	curr_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
	if source == None:
		print(f"[{curr_time}]  : {text}")
	else :
		print(f"[{curr_time}]  : ({source}) - {text}")


"""
 Gets the id of which shutter is open in the slitlet. If SHUTSTA is unusual (i.e. only one shutter is open), returns None
"""
def WhichShutterOpen(hdr):
	_ = hdr["SHUTSTA"]
	if _ == "11x":
		return 2
	elif _ == "1x1":
		return 1
	elif _ == "x11":
		return 0
	else:
		return None


"""
Simple gaussian function
"""
def gaussian(x,x0,A,s):
	return A * np.exp(-(x-x0)**2 / (2*s**2))

"""
Slitlet model : 3 gaussians of same sigma
"""
def slitletModel(x,x1,x2,x3,A1,A2,A3,s):
	return gaussian(x,x1,A1,s) + gaussian(x,x2,A2,s) + gaussian(x,x3,A3,s)


def IDWExtrapolation(xy, ui, power=1):
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

		w = d**(-power/2)

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
																												"_bkg")
			if not data["products"][0]["members"][i]["exptype"] == "science":
				not_science.append(i)

		# Starting from the end in order to keep the values at the same index
		for i in not_science[::-1]:
			del data["products"][0]["members"][i]

	with open(file, "w") as asn:
		json.dump(data, asn, indent=4)