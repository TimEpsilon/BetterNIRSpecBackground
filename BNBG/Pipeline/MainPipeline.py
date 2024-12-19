import json
import os

from jwst.extract_1d import Extract1dStep

from BNBG.utils import getCRDSPath

os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline, Spec3Pipeline
from jwst.pipeline import Spec2Pipeline
import stdatamodels.jwst.datamodels as dm
from jwst.wavecorr import WavecorrStep
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.pixel_replace import PixelReplaceStep
from jwst.resample import ResampleSpecStep

from BNBG.utils import logConsole, rewriteJSON, DoPipelineWithCheckpoints
import BNBG.Pipeline.BetterBackgroundSubtractStep as BkgSubtractStep


def Stage1(uncal,path):
	"""
	Applies the first stage of the pipeline. No notable modifications to the steps are made
	Parameters
	----------
	uncal : path to the file
	path : path to the folder

	"""
	logConsole(f"Starting Stage 1 on {os.path.basename(uncal)}")
	if os.path.exists(uncal.replace("_uncal", "_rate")):
		logConsole(f"File {os.path.basename(uncal).replace('uncal','rate')} already exists, skipping.")
		return
	det1 = Detector1Pipeline()
	det1.save_results = True
	det1.output_dir = path
	det1.run(uncal)
	del det1
	return


def Stage2(rate, path):
	"""
	Applies the second stage of the pipeline. This used to be where the custom subtraction happened.
	This now happens after the stage 3 si finished.
	Parameters
	----------
	rate : path to the file
	path : path to the folder
	"""
	logConsole(f"Starting Stage 2 on {os.path.basename(rate)}")
	if os.path.exists(rate.replace("_rate.fits", "_cal.fits")):
		logConsole(f"File {os.path.basename(rate).replace('rate','cal')} already exists, skipping.")
		return

	# No background subtraction
	steps = {'master_background_mos': {'skip': True},
			 'bkg_subtract': {'skip': True}}

	spec2 = Spec2Pipeline(steps=steps)
	spec2.output_dir = path
	spec2.run(rate)
	del spec2

def Stage2Default(rate, path):
	"""
	Applies the second stage of the pipeline as usual.
	Parameters
	----------
	rate : should be the path to a spec2.json
	path
	"""
	logConsole(f"Starting Basic Stage 2 (Default)")

	with open(rate) as file:
		_ = json.load(file)

	calFile = _["products"][0]["name"] + "_cal.fits"

	if os.path.exists(os.path.join(path, calFile)):
		logConsole("File already exists, skipping...")
		return

	spec2 = Spec2Pipeline()
	spec2.output_dir = path
	spec2.save_results = True
	spec2.run(rate)
	del spec2

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
	pathBNBG = s2d.replace("_s2d.fits", "_s2d-BNBG.fits")
	if not os.path.exists(pathBNBG):
		step = BkgSubtractStep.BetterBackgroundStep()
		s2d_BNBG = step.call(pathBNBG, output_dir=os.path.dirname(path))
		x1dStep = Extract1dStep()
		x1dStep.suffix = "x1d-BNBG"
		x1dStep.save_results = True
		x1dStep.output_dir = path
		x1dStep.run(s2d_BNBG)

