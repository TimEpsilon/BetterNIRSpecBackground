import json
import os

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

from BNBG.utils import logConsole, rewriteJSON, numberSameLength
import BNBG.Pipeline.BetterBackgroundSubtractStep as BkgSubtractStep


def Stage1(uncal,path):
	"""
	Applies the first stage of the pipeline. No notable modifications to the steps are made
	Parameters
	----------
	uncal : path to the file
	path : path to the folder

	"""
	logConsole(f"Starting Stage 1 on {uncal}")
	if os.path.exists(uncal.replace("_uncal", "_rate")):
		return
	det1 = Detector1Pipeline()
	det1.save_results = True
	det1.output_dir = path
	det1.run(uncal)
	del det1
	return


def Stage2(rate,path):
	"""
	Applies the second stage of the pipeline. It is applied in 3 steps :
	- The first few steps of the pipeline are applied as usual up until srctype
	- The pipeline is stopped and the custom step is applied
	- The remaining steps are applied
	Parameters
	----------
	rate
	path
	"""
	logConsole(f"Starting Stage 2")
	pathSrctype = rate.replace("_rate.fits", "_srctype.fits")
	if not os.path.exists(pathSrctype):
		steps = {'srctype': {'save_results': True},
				 'photom': {'skip': True},
				 'flat_field': {'skip': True},
				 'master_background_mos': {'skip': True},
				 'wavecorr': {'skip': True},
				 'pathloss': {'skip': True},
				 'barshadow': {'skip': True},
				 'pixel_replace': {'skip': True},
				 'extract_1d': {'skip': True},
				 'cube_build': {'skip': True},
				 'resample_spec': {'skip': True}}

		spec2 = Spec2Pipeline(steps=steps)
		spec2.output_dir = path
		spec2.run(rate)
		del spec2

	# Custom Step
	pathBNBG = rate.replace("_rate.fits", "_BNBG.fits")
	if not os.path.exists(pathBNBG):
		step = BkgSubtractStep.BetterBackgroundStep()
		step.call(pathSrctype,output_dir=os.path.dirname(pathSrctype))

	pathPhotom = rate.replace("_rate.fits", "_BNBG_photomstep.fits")
	if not os.path.exists(pathPhotom):
		logConsole("Restarting Pipeline Stage 2")

		# Steps :
		# wavecorr
		# flat field
		# path loss
		# bar shadow
		# photom
		# pixel replace
		# rectified 2D -> Save
		# spectral extraction -> Save

		# Remaining Steps
		with dm.open(pathBNBG) as data:
			logConsole("Successfully loaded _BNBG file")
			calibrated = WavecorrStep.call(data)
			calibrated = FlatFieldStep.call(calibrated)
			calibrated = PathLossStep.call(calibrated)
			calibrated = BarShadowStep.call(calibrated)
			calibrated = PhotomStep.call(calibrated, output_dir=path, save_results=True)
			calibrated = PixelReplaceStep.call(calibrated)
			calibrated = ResampleSpecStep.call(calibrated, output_dir=path, save_results=True)
			#calibrated = Extract1dStep.call(calibrated, output_dir=path, save_results=True)
			del calibrated


def Stage2NoSubtraction(rate, path):
	"""
	Applies the second stage of the pipeline.
	This applies the steps as if it was the custom pipeline, but skips the subtraction part
	Parameters
	----------
	rate
	path
	"""
	logConsole(f"Starting Stage 2 (No Subtraction)")
	pathSrctype = rate.replace("_rate.fits", "_srctype.fits")
	if not os.path.exists(pathSrctype):
		steps = {'srctype': {'save_results': True},
				 'photom': {'skip': True},
				 'flat_field': {'skip': True},
				 'master_background_mos': {'skip': True},
				 'wavecorr': {'skip': True},
				 'pathloss': {'skip': True},
				 'barshadow': {'skip': True},
				 'pixel_replace': {'skip': True},
				 'extract_1d': {'skip': True},
				 'cube_build': {'skip': True},
				 'resample_spec': {'skip': True}}

		spec2 = Spec2Pipeline(steps=steps)
		spec2.output_dir = path
		spec2.run(rate)
		del spec2

	name = os.path.basename(rate.replace("_rate.fits", "_photomstep.fits"))
	pathPhotom = os.path.join(path, name)
	if not os.path.exists(pathPhotom):
		logConsole("Restarting Pipeline Stage 2 (No Subtraction)")

		# Remaining Steps
		with dm.open(pathSrctype) as data:
			logConsole("Successfully loaded _BNBG file")
			calibrated = WavecorrStep.call(data)
			calibrated = FlatFieldStep.call(calibrated)
			calibrated = PathLossStep.call(calibrated)
			calibrated = BarShadowStep.call(calibrated)
			calibrated = PhotomStep.call(calibrated, output_dir=path, save_results=True)
			calibrated = PixelReplaceStep.call(calibrated)
			calibrated = ResampleSpecStep.call(calibrated, output_dir=path, save_results=True)
			del calibrated

def Stage2Default(rate, path):
	"""
	Applies the second stage of the pipeline as usual.
	Parameters
	----------
	rate : should be the path to a spec2.json
	path
	"""
	logConsole(f"Starting Basic Stage 2 (Default)")

	with open("test.json") as file:
		_ = json.load(file)

	calFile = _["products"][0]["name"] + "_cal.fits"

	if os.path.exists(os.path.join(path, calFile)):
		logConsole("File already exists, skipping...")

	spec2 = Spec2Pipeline()
	spec2.output_dir = path
	spec2.save_results = True
	spec2.run(rate)
	del spec2

def Stage3_AssociationFile(asn_list, path, suffix="_BNBG_photomstep"):
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
