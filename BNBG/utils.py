import json
import logging
import os
import stdatamodels.jwst.datamodels as dm
import jwst.lib.suffix as sufx

# Update suffixes
suffixList = ["bkg-BNBG", "cal-BNBG"]
sufx.SUFFIXES_TO_ADD += suffixList
sufx.KNOW_SUFFIXES = sufx.combine_suffixes()

# Get logger of jwst pipeline
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


def rewriteJSON(file, suffix="BNBG_photomstep"):
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
			name = data["products"][0]["members"][i]["expname"]
			# The name of the file will be as such : jw01345063001_03101_00001_nrs1_BNBG_photomstep.fits
			# We can't simply use _ as delimiters because of BNBG_something
			# We simply get rid of the .fits and every character after nrs1/nrs2
			# This is a janky solution, but that's the cost of having multiple checkpoint files I suppose
			if "nrs1" in name:
				name = f"{name.split('nrs1')[0]}nrs1_{suffix}.fits"
			if "nrs2" in name:
				name = f"{name.split('nrs2')[0]}nrs2_{suffix}.fits"
			data["products"][0]["members"][i]["expname"] = name

			if not data["products"][0]["members"][i]["exptype"] == "science":
				not_science.append(i)

		# Starting from the end in order to keep the values at the same index
		for i in not_science[::-1]:
			del data["products"][0]["members"][i]

	with open(file, "w") as asn:
		json.dump(data, asn, indent=4)

def getCRDSPath():
	# Get the directory of the current script
	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, 'CRDS_PATH.txt')
	with open(path) as file:
		txt = file.readline()
		if txt is None or txt == "":
			raise Exception("CRDS_PATH.txt not found or file empty")
		return txt

# Will keep this in case it is needed at one point
# However, it is currently useless as the Stage2 Pipeline is hardcoded to stop if assign_wcs is skipped
# Which is a problem if I want to use any kind of checkpoint file
# I can't be the only one who wants to save the results at different points in the pipeline and reuse them later
# Why is there no easy way to do this
# This should be included within the pipeline itself in the first place
def DoPipelineWithCheckpoints(pipeline, file):
	"""
	Applies a pipeline stage to a file which could have been generated at any step within the pipeline, or a so-called checkpoint file
	Parameters
	----------
	pipeline : Pipeline
		Will be applied to the file, but every step already applied or skipped will be skipped
	file :
		A path to a file, either an ASN json or a fits file, or anything else that can be opened as or is a datamodel
	"""
	ASN = None
	if isinstance(file, str):
		if ".json" in file:
			# Keep the 1st file within the json
			logConsole("Found a json file. Will grab the first fits file found in the association.")
			with open(file) as jsonFile:
				_ = json.load(jsonFile)

			name = _["products"][0]["members"][0]["expname"]
			ASN = file
			file = os.path.join(os.path.dirname(file), name)

	model = dm.open(file)

	# Iterate over steps in the pipeline
	for stepName, _ in pipeline.step_defs.items():
		thisStep = getattr(pipeline, stepName)
		# Check if this step has already been completed or skipped
		if stepName in dir(model.meta.cal_step):
			if getattr(model.meta.cal_step, stepName) in ['COMPLETE', 'SKIPPED']:
				logConsole(f"{stepName} has already been applied. Will be skipped.")
				thisStep.skip = True
			else:
				thisStep.skip = False

	if ASN is not None:
		pipeline.run(ASN)
	else:
		pipeline.run(model)

def getSourcePosition(slit):
	"""
	Returns the vertical position, in detector space (pixels), of the source
	Parameters
	----------
	slit : a slit object, it is assumed that assign_wcs has been applied beforehand

	Returns
	-------
	vertical source position in pixels, can be ~1e48 if the source is outside in the case of a 2 slit slitlet
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

	def withSuffix(self, suffix : str):
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