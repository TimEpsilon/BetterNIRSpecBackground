import os
import time

from BNBG.utils import getCRDSPath, PathManager

os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline, Spec3Pipeline, Spec2Pipeline
from jwst.extract_1d import Extract1dStep
from stdatamodels.jwst import datamodels as dm

from BNBG.utils import logConsole, rewriteJSON
import BNBG.Pipeline.BetterBackgroundSubtractStep as BkgSubtractStep


def Stage1(uncal,path):
	"""
	Applies the first stage of the pipeline. No notable modifications to the steps are made
	Parameters
	----------
	uncal : path to the file
	path : path to the folder

	"""
	uncalPath = PathManager(uncal)
	logConsole(f"Starting Stage 1 on {uncalPath.filename}")

	def pipe1():
		det1 = Detector1Pipeline()
		det1.save_results = True
		det1.output_dir = path
		det1.run(uncal)
	# Handles the logic of checkpoint files, i.e. will apply the pipeline only if output file can't be found
	uncalPath.openSuffix("rate", pipe1, open=False)

def Stage2(asn, path):
	"""
	Applies the second stage of the pipeline. This is where the custom subtraction happens, after the pipeline has run.
	Parameters
	----------
	asn : path to the file
	path : path to the folder
	"""
	ratePath = PathManager(asn)
	logConsole(f"Starting Stage 2 on {ratePath.filename}")

	def pipe2():
		# No background subtraction
		steps = {'master_background_mos': {'skip': True},
				 'bkg_subtract': {'skip': True}}
		spec2 = Spec2Pipeline(steps=steps)
		spec2.output_dir = path
		spec2.save_results = True
		cal = spec2.run(asn)
		return cal

	cal = ratePath.openSuffix("cal", pipe2)
	s2d = dm.open(ratePath.withSuffix("s2d"))  # The pipeline also saves a resampled file, which we need for the background.

	# TODO : custom subtraction

def Stage3_AssociationFile(asn_list, path, suffix="cal"):
	# Create a separate folder for all final data
	final = path + "Final/"
	if not os.path.exists(final):
		os.makedirs(final)

	finishedFile = os.path.join(final, "finished")
	if os.path.exists(finishedFile):
		logConsole("Folder has already been processed")
		return

	for asn in asn_list:
		logConsole(f"Starting Stage 3")
		logConsole("Modifying Stage 3 association files")
		rewriteJSON(asn, suffix=suffix)

		spec3 = Spec3Pipeline()
		spec3.save_results = True
		spec3.output_dir = final
		spec3.run(asn)
		del spec3

	# Used to determine if the code has been run
	# Delete this file if you want the stage 3 to happen again
	finishedFile = open(final+"finished", "w")
	finishedFile.write("")
	finishedFile.close()

def Stage4(s2d, path):
	startTime = time.time()
	name = os.path.basename(s2d)
	pathBNBG = path + name.replace("_s2d.fits", "_s2d-BNBG.fits")
	logConsole(f"Starting work on {name}")

	if not os.path.exists(pathBNBG):
		s2d_BNBG = BkgSubtractStep.BetterBackgroundStep(s2d, path)
		x1dStep = Extract1dStep()
		x1d = x1dStep.run(s2d_BNBG)
		x1d.save(pathBNBG.replace("_s2d", "_x1d"))
	endTime = time.time()
	logConsole(f"Finished in {round(endTime - startTime, 3)}s")

