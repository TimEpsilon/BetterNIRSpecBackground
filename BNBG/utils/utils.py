import json
import os
import re

import numpy as np
import stdatamodels.jwst.datamodels as dm
from scipy.optimize import curve_fit
from stdatamodels.jwst.datamodels import SlitModel
import jwst.lib.suffix as sufx

from .logger import logConsole

# Update suffixes
suffixList = ["bkg-BNBG", "cal-BNBG"]
sufx.SUFFIXES_TO_ADD += suffixList
sufx.KNOW_SUFFIXES = sufx.combine_suffixes()
sufx.REMOVE_SUFFIX_REGEX = re.compile(
    '^(?P<root>.+?)((?P<separator>_|-)(' +
    '|'.join(sufx.KNOW_SUFFIXES) + '))?$'
)

def rewriteJSON(file, suffix="cal-BNBG"):
	"""
	Rewrites the asn.json files in order to apply to the _BNBG files

	Parameters
	----------
	suffix : str
		Suffix of the input fits files

	file : str
		Path to the asn.json file
	"""

	with open(file, "r") as asn:
		data = json.load(asn)

		not_science = [] # Removes other type of files, like images
		for i in range(len(data["products"][0]["members"])):
			name = data["products"][0]["members"][i]["expname"].split(".")[0]
			name = sufx.remove_suffix(name)[0]
			name = name + "_" + suffix + ".fits"
			data["products"][0]["members"][i]["expname"] = name

			if not data["products"][0]["members"][i]["exptype"] == "science":
				not_science.append(i)

		# Starting from the end in order to keep the values at the same index
		for i in not_science[::-1]:
			del data["products"][0]["members"][i]

	# Overwrite original file
	with open(file, "w") as asn:
		json.dump(data, asn, indent=4)

def getSourcePosition(slit : SlitModel) -> float | None:
	"""
	Returns the vertical position, in detector space (pixels), of the source.
	Will try to approximate the highest spatial peak as the source.
	If however such a peak can't be found, will default to the approximate source position given by the pipeline.
	If somehow the pipeline returns nothing, will return None.

	Parameters
	----------
	slit : SlitModel
		Single slit datamodel

	Returns
	-------
	position : float
		Vertical source position in pixels, can be ~1e48 if the source is outside in the case of a 2 slit slitlet
	"""
	source = slit.meta.wcs.transform('world', 'detector', slit.source_ra, slit.source_dec, 3)[1]

	# Somehow this can happen
	if np.isnan(source):
		return None

	data = slit.data.copy()
	# Quick cleanup of negatives + crop
	Y, _ = np.indices(data.shape)
	mask = (data <= 0) | (Y < 3) | (Y > Y.max() - 3) | np.isnan(data)
	data[mask] = np.nan

	# Spatial distribution
	distribution = np.nanmedian(data, axis=1)
	if 0 <= source < len(distribution):
		peak = distribution[int(source)]
	else :
		peak = np.nanmax(distribution)

	X = np.indices(distribution.shape)[0]
	mask = np.isfinite(distribution)
	X = X[mask]
	distribution = distribution[mask]

	try :
		coeff, err = curve_fit(lambda x, x0, s, A, c : A*np.exp(-(x-x0)**2/(2*s**2))+c,
						  X,
						  distribution,
						  p0=[source, 3, peak, np.min(distribution)],
						  bounds=([source-4, 0.01, 0, np.min(distribution)],
								  [source+4, 10, np.max(distribution), np.max(distribution)]))

		# We approximate the SNR as (A+c)/c = A/c + 1, and we only keep A/c > 0.8
		# We also exclude fits where the uncertainty on the position is too large
		snr = coeff[2]/coeff[3]
		if snr < 0.8 or err[1][1] > data.shape[1] / 2:
			return source
		else:
			return coeff[0]

	except Exception as e:
		logConsole("Can't find a suitable fit for source position. Defaulting to approximate position.")
		return source

class PathManager :
	def __init__(self, path : str):
		"""
		Custom Class for managing paths output from the pipeline. If the path is a json, will be treated as an association,
		i.e. will get the 1st science file within the association and use that instead.

		Parameters
		----------
		path : str
			Path to a file. This serves as a basis for constructing similar filenames.
		"""
		path = os.path.normpath(path)

		self.dirname : str = os.path.dirname(path)  # Name of directory

		# In case of json asn file
		if os.path.splitext(path)[1] == ".json":
			# Grabs the first science file
			with open(path) as jsonFile:
				_ = json.load(jsonFile)
				members = _["products"][0]["members"]
				members = [_ for _ in members if _["exptype"] == "science"]
			filename = members[0]["expname"]
		else :
			filename = os.path.basename(path)

		self.filename = filename # Full filename
		self.root, self.extension = os.path.splitext(self.filename) # True name of file + extension of file (should be .fits)
		self.name = sufx.remove_suffix(self.root)[0] # Base name of file, free of suffixes

	def withSuffix(self, suffix : str) -> str:
		"""
		Returns the path with a new suffix appended.

		Parameters
		----------
		suffix : str
			A string to be appended to the file as "_suffix"
		"""
		return os.path.join(self.dirname, self.name + "_" + suffix + self.extension)

	def openSuffix(self, suffix : str, function : callable, open=True):
		"""
		Returns a datamodel for a given suffix if it exists, else executes a function which returns this datamodel.
		Useful for checkpoint files.

		Parameters
		----------
		suffix : str
			A string to be appended to the file as "_suffix"

		function : callable
			A function to execute when no file is found. Should return a datamodel, saved as the newPath, and take no arguments.
			If the function has arguments, simply pass lambda : function(args)

		open : boolean
			If this should return a datamodel. Allows for skipping long loading times for large models.
		"""
		newPath = self.withSuffix(suffix)
		if os.path.exists(newPath):
			target = self.name + "_" + suffix + self.extension
			logConsole(f"File {target} already exists. Skipping...")
			if open:
				return dm.open(newPath)
		else:
			return function()