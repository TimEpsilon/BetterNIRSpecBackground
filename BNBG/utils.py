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

def getCRDSPath(path="../CRDS_PATH.txt"):
	with open(path) as file:
		txt = file.readline()
		if txt is None or txt == "":
			raise Exception("CRDS_PATH.txt not found or file empty")
		return txt


def verifySimilarImages(A, B):
	"""
	Verifies if 2 images A and B are both valid ndarrays and have the same shape

	Parameters
	----------
	A : ndarray
	B : ndarray
	"""
	return isinstance(A, np.ndarray) and isinstance(B, np.ndarray) and A.shape == B.shape