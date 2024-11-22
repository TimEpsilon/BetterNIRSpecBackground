import os
from glob import glob

import pandas as pd
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
from jwst.extract_1d import Extract1dStep

from BNBG.utils import logConsole, rewriteJSON, numberSameLength
import BetterBackgroundSubtractStep as BkgSubtractStep


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
	if not os.path.exists(rate.replace("rate", "srctype")):
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
	pathSrctype = rate.replace("_rate", "_srctype")
	if not os.path.exists(pathSrctype):
		step = BkgSubtractStep.BetterBackgroundStep()
		step.call(pathSrctype,output_dir=os.path.dirname(pathSrctype))

	pathBNBG = rate.replace("_rate", "_BNBG")
	pathPhotom = rate.replace("_rate", "_photomstep")

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
			#calibrated = PixelReplaceStep.call(calibrated)
			#calibrated = ResampleSpecStep.call(calibrated, output_dir=path, save_results=True)
			#calibrated = Extract1dStep.call(calibrated, output_dir=path, save_results=True)
			del calibrated

def Stage3_AssociationFile(asn_list, path):
	for asn in asn_list:
		logConsole(f"Starting Stage 3")
		logConsole("Modifying Stage 3 association files")
		rewriteJSON(asn)

		final = path + "Final/"
		if not os.path.exists(final):
			os.makedirs(final)

		spec3 = Spec3Pipeline()
		spec3.save_results = True
		spec3.output_dir = final
		spec3.run(asn)
		del spec3

def Stage3_FinishUp(path):
	"""
	Creates a file signifying that the pipeline has finished
	As a bonus, this file acts as a table containing the names of the files mentioned in slits_with_double_object.dat
	Parameters
	----------
	path : path to the folder
	"""

	double_slits = pd.read_csv("../slits_with_double_object.dat", sep=",")
	main_target = double_slits["Central_target"]
	companion = double_slits["Companion"]
	main_target = main_target.apply(numberSameLength)
	companion = companion.apply(numberSameLength)

	target_path = []
	n = len(main_target)
	for i in range(n):
		target = main_target[i]
		_ = glob(f"{path}*{target}*_s2d.fits")
		if len(_) > 0 and f"P{double_slits['Pointing'][i]}" in _[0]:
			target_path.append(_[0].split("/")[-1])
		else:
			target_path.append(None)

	file_of_interest = {"TargetType": ["Main"] * len(main_target) + ["Companion"] * len(main_target),
						"ID": [*main_target, *companion],
						"Path": target_path}

	df = pd.DataFrame(file_of_interest)
	df.to_csv(f"{path}FilesOfInterest.csv")