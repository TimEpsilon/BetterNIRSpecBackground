#  Copyright (c) 2025. Tim Dewachter, LAM

import os

from BNBG.Pipeline.BetterBackgroundSubtractStep import BetterBackgroundStep
from ..utils.CrdsSetup import getCRDSPath

os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from ..utils.utils import PathManager # Later import to avoid CRDS shenanigans

from jwst.pipeline import Detector1Pipeline, Spec3Pipeline, Spec2Pipeline
from stdatamodels.jwst import datamodels as dm

from ..utils.utils import rewriteJSON
from ..utils.logger import logConsole


def Stage1(uncal : str,path : str):
	"""
	Applies the first stage of the pipeline. No notable modifications to the steps are made
	Parameters
	----------
	uncal : str
		Path to the file

	path : str
		Path to the folder

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

def Stage2(asn : str, path : str, skipbnbg=False, **kwargs):
	"""
	Applies the second stage of the pipeline. This is where the custom subtraction happens, after the pipeline has run.
	Parameters
	----------
	asn : str
		Path to the _spec2 asn file. Should work as a single fits but not recommended

	path : str
		Path to the folder

	skipbnbg : bool
		Whether to skip BNBG or not. Defaults to False

	**kwargs :
		Arguments to pass to BetterBackgroundStep

	"""
	ratePath = PathManager(asn)
	logConsole(f"Starting Stage 2 on {ratePath.filename}")

	if os.path.exists(ratePath.withSuffix("cal-BNBG")):
		return

	def pipe2():
		# No background subtraction
		# Since the goal is to detect secondary sources, point sources will be treated as extended
		steps = {'master_background_mos': {'skip': True},
				 'bkg_subtract': {'skip': True},
				 'extract_1d': {'skip': True},
				 "pathloss": {"source_type": "EXTENDED"}, # Spectral correction to be independent of spatial position
				 "barshadow": {"source_type": "EXTENDED"} # (Almost) gets rid of envelope around each shutter
				 }
		spec2 = Spec2Pipeline(steps=steps)
		spec2.output_dir = path
		spec2.save_results = True
		_ = spec2.run(asn)
		return _[0]

	cal = ratePath.openSuffix("cal", pipe2)
	logConsole(f"Opening corresponding s2d file...")
	s2d = dm.open(ratePath.withSuffix("s2d"))  # The pipeline also saves a resampled file, which we need for the background.

	if not skipbnbg:
		step = BetterBackgroundStep(ratePath, s2d, cal, **kwargs)
		step()
	else:
		logConsole(f"Skipping BNBG...")

def Stage3(asn : str, path : str, suffix="cal-BNBG"):
	"""
	Applies the last stage of the pipeline.
	Parameters
	----------
	asn : str
		Path to the _spec3 asn file

	path : str
		Path to the folder

	suffix : str
		The suffix to which files will be changed in the association json; i.e. the suffix of input files.

	"""

	# Create a separate folder for all final data
	finalDirectory = os.path.join(path,"Final/")
	if not os.path.exists(finalDirectory):
		os.makedirs(finalDirectory)

	# Skip folder if finished file already exists
	# This is an empty file
	finishedFile = os.path.join(finalDirectory, "finished")
	if os.path.exists(finishedFile):
		logConsole("Folder has already been processed")
		return

	logConsole(f"Starting Stage 3 on {os.path.basename(asn)}")
	rewriteJSON(asn, suffix=suffix)

	# No PathManager here because filename changes
	spec3 = Spec3Pipeline()
	spec3.save_results = True
	spec3.output_dir = finalDirectory
	spec3.run(asn)

	# Used to determine if the code has been run
	# Delete this file if you want the stage 3 to happen again
	finishedFile = open(finalDirectory+"finished", "w")
	finishedFile.write("")
	finishedFile.close()

