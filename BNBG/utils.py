import json
import logging
import os

import stdatamodels.jwst.datamodels as dm
from stdatamodels.jwst.datamodels import SlitModel
import jwst.lib.suffix as sufx

# Update suffixes
suffixList = ["bkg-BNBG", "cal-BNBG"]
sufx.SUFFIXES_TO_ADD += suffixList
sufx.KNOW_SUFFIXES = sufx.combine_suffixes()

# Get logger of jwst pipeline
logger = logging.getLogger("stpipe")

def logConsole(text : str, source=None):
	"""
	 Logger : displays time + log

	Parameters
	----------
	text : str
		Text to display
	source : str
		 WARNING, ERROR, DEBUG, INFO/None
	"""
	text = f" - [BetterBackground]  : {text}"
	logType = {"WARNING": lambda : logger.warning(text),
			   "ERROR": lambda : logger.error(text),
			   "DEBUG": lambda : logger.debug(text)
			   }

	logType.get(source,lambda : logger.info(text))()

def rewriteJSON(file, suffix="BNBG_photomstep"):
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
			name = data["products"][0]["members"][i]["expname"]
			name = sufx.remove_suffix(name)
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

def getCRDSPath() -> str:
	"""
	Get the path to the CRDS file from "./CRDS_PATH.txt"

	Returns
	-------
	txt : str
		Absolute path to the CRDS file

	"""
	# Get the directory of the current script
	scriptDirectory = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(scriptDirectory, 'CRDS_PATH.txt')
	with open(path) as file:
		txt = file.readline()
		if txt is None or txt == "":
			raise Exception("CRDS_PATH.txt not found or file empty")
		logConsole(f"CRDS folder at {txt}")
		return txt

def getSourcePosition(slit : SlitModel) -> float:
	"""
	Returns the vertical position, in detector space (pixels), of the source

	Parameters
	----------
	slit : SlitModel
		Single slit datamodel

	Returns
	-------
	position : float
		Vertical source position in pixels, can be ~1e48 if the source is outside in the case of a 2 slit slitlet
	"""
	return slit.meta.wcs.transform('world', 'detector', slit.source_ra, slit.source_dec, 3)[1]

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

		self.dirname = os.path.dirname(path)  # Name of directory

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