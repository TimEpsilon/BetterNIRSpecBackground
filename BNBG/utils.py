import numpy as np
import json
import logging
import os
import stdatamodels.jwst.datamodels as dm


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
			name = data["products"][0]["members"][i]["expname"]
			# The name of the file will be as such : jw01345063001_03101_00001_nrs1_BNBG_photomstep.fits
			# We can't simply use _ as delimiters because of BNBG_something
			# We simply get rid of the .fits and every character after nrs1/nrs2
			# This is a janky solution, but that's the cost of having multiple checkpoint files I suppose
			if "nrs1" in name:
				print(f"{name.split('nrs1')[0]}nrs1_{suffix}.fits")
			if "nrs2" in name:
				print(f"{name.split('nrs2')[0]}nrs2_{suffix}.fits")
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

def verifySimilarImages(A, B):
	"""
	Verifies if 2 images A and B are both valid ndarrays and have the same shape

	Parameters
	----------
	A : ndarray
	B : ndarray
	"""
	return isinstance(A, np.ndarray) and isinstance(B, np.ndarray) and A.shape == B.shape

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