import os

os.environ['CRDS_PATH'] = '/home/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline
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

from glob import glob
import BetterBackgroundSubtractStep as BkgSubtractStep

import utils
from utils import logConsole


path = "./detailedPipeline/"


########
# This is a simplified copy of apply_pipeline.py
# The goal of this file is to apply the pipeline to a chosen file and to save every step of the pipeline
# This serves to analyse the effect of each step on the files
# Only the 1st _uncal file will be used, the rest will be ignored
########


##########
# Stage 1
##########


uncal = glob(path + "*_uncal.fits")[0]

logConsole(f"Starting Stage 1 on {uncal}")

if not os.path.exists(uncal.replace("_uncal", "_rate")):

	# Steps Params By Pablo Arrabal Haro
	steps = {
			'jump': {'expand_large_events': True,
					 # 1st flag groups after jump above DN threshold.
					 'after_jump_flag_dn1': 0,
					 # 1st flag groups after jump groups within
					 # specified time.
					 'after_jump_flag_time1': 0,
					 # 2nd flag groups after jump above DN threshold.
					 'after_jump_flag_dn2': 0,
					 # 2nd flag groups after jump groups within
					 # specified time.
					 'after_jump_flag_time2': 0,
					 # Minimum required area for the central saturation
					 # of snowballs.
					 'min_sat_area': 15.0,
					 # Minimum area to trigger large events processing.
					 'min_jump_area': 15.0,
					 # The expansion factor for the enclosing circles
					 # or ellipses.
					 'expand_factor': 2.0,
					 'save_results' : True},
			'group_scale' : {'save_results' : True},
			'dq_init' : {'save_results' : True},
			'saturation': {'save_results' : True},
			'superbias': {'save_results' : True},
			'refpix': {'save_results' : True},
			'linearity': {'save_results' : True},
			'persistence': {'save_results' : True},
			'dark_current': {'save_results' : True},
			'charge_migration': {'save_results' : True},
			'ramp_fit': {'save_results' : True},
			'gain_scale': {'save_results' : True}
			}
	det1 = Detector1Pipeline(steps=steps)
	det1.save_results = True
	det1.output_dir = path
	det1.run(uncal)

	det1 = None

##########
# Stage 2
##########

logConsole(f"Stage 1 Finished. Preparing Stage 2")

rate = glob(path+"*_rate.fits")[0]
logConsole(f"Starting Stage 2 on {rate}")

if not os.path.exists(rate.replace("rate","srctype")):
		steps={'srctype': {'save_results':True},
				'photom': {'skip':True},
				'flat_field': {'skip':True},
				'master_background_mos': {'skip':True},
				'wavecorr': {'skip':True},
				'pathloss': {'skip':True},
				'barshadow': {'skip':True},
				'pixel_replace': {'skip':True},
				'extract_1d': {'skip':True},
				'cube_build': {'skip':True},
				'resample_spec': {'skip':True}}

		spec2 = Spec2Pipeline(steps=steps)
		spec2.output_dir = path
		spec2.run(rate)

if not os.path.exists(rate.replace("rate","bkg")):
	BkgSubtractStep.BetterBackgroundStep(rate.replace("_rate", "_srctype"))

bkg = rate.replace("_rate","_bkg")
spec2 = None

if not os.path.exists(bkg.replace("_bkg","_cal")):
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

	with dm.open(bkg) as data:
		logConsole("Successfully loaded _bkg file")
		calibrated = WavecorrStep.call(data,save_results=True)
		calibrated = FlatFieldStep.call(calibrated,save_results=True)
		calibrated = PathLossStep.call(calibrated,save_results=True)
		calibrated = BarShadowStep.call(calibrated,save_results=True)
		calibrated = PhotomStep.call(calibrated,output_dir=path,save_results=True)
		calibrated = PixelReplaceStep.call(calibrated,save_results=True)
		calibrated = ResampleSpecStep.call(calibrated,output_dir=path,save_results=True)
		calibrated = Extract1dStep.call(calibrated,output_dir=path,save_results=True)

