import numpy as np
import json
import logging
import os

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


def rewriteJSON(file, suffix="_BNBG_photomstep"):
	"""
	Rewrites the asn.json files in order to apply to the _BNBG files

	Parameters
	----------
	suffix : str
	Suffix of the input fits files

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
			data["products"][0]["members"][i]["expname"] = data["products"][0]["members"][i]["expname"].replace("_cal",suffix)

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

def getCRDSPath():
	# Get the directory of the current script
	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, 'CRDS_PATH.txt')
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