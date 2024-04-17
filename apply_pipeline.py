import os

os.environ['CRDS_PATH'] = '/home/tim/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline
from glob import glob
import BetterBackgroundSubstractStep as BkgSubstractStep
import numpy as np


from utils import logConsole


working_dir = "./mastDownload/JWST/"

#########
# Monkey Patch Master Background method
#########

#BkgMosStep.process = BetterMasterBackgroundMosStep
#logConsole("Overriding Master Background method")


logConsole(f"Found {len(os.listdir(working_dir))} folders")
for folder in os.listdir(working_dir):
	break
	path = working_dir + folder + "/"
	logConsole(f"Starting on {folder}")

	##########
	# Stage 1
	##########


	uncal_list = glob(path+"*_uncal.fits")
	logConsole(f"Found {len(uncal_list)} uncalibrated files")

	for n,uncal in enumerate(uncal_list):
		logConsole(f"Starting Stage 1 ({n+1}/{len(uncal_list)})")
		if os.path.exists(uncal.replace("_uncal","_rate")):
			continue
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
                         'expand_factor': 2.0}
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

for folder in os.listdir(working_dir):
	path = working_dir + folder + "/"
	logConsole(f"Starting on {folder}")

	rate_list = glob(path+"*_rate.fits")
	logConsole(f"Found {len(rate_list)} countrate files")

	for n,rate in enumerate(rate_list):
		logConsole(f"Starting Stage 2 ({n+1}/{len(rate_list)})")
		if ~os.path.exists(rate.replace("rate","srctype")):
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

		if ~os.path.exists(rate.replace("rate","bkg")):
			BkgSubstractStep.BetterBackgroundStep(rate.replace("_rate","_srctype"))

		bkg = rate.replace("_rate","_bkg")

		if ~os.path.exists(bkg.replace("bkg","cal")):
			steps = {
				"assign_wcs" : {'skip':True},
				"msa_flagging" : {'skip':True},
				"extract_2d" : {'skip':True},
				"srctype" : {'skip':True},
				'master_background_mos': {'skip': True}
			}

			spec2 = Spec2Pipeline(steps=steps)
			spec2.save_results = True
			spec2.output_dir = path
			spec2.run(bkg)

		spec2 = None


	##########
	# TODO : Master Background
	# 0 - Apply Spec2 up until srctype (included) and save them
	# 1 - Modify the _srctype files, all spectra
	# 2 - Create a _bkg file, exact copy
	# 3 - Modify the spec2_asn.json files in order to apply to those _bkg
	# 4 - Apply the rest of Spec2 to spec2_asn.json
	# 5 - Apply Spec3 to the spec3_asn.json
	##########

	exit()

	##########
	# Stage 3
	##########

	logConsole(f"Stage 2 Finished. Preparing Stage 3")

for folder in os.listdir(working_dir):
	path = working_dir + folder + "/"
	logConsole(f"Starting on {folder}")

	asn_list = glob(path+"*_spec2_*_asn.json")
	logConsole(f"Found {len(asn_list)} association files")

	for n,asn in enumerate(asn_list):
		logConsole(f"Starting Stage 3 ({n+1}/{len(asn_list)})")
		Spec3Pipeline.call(asn,save_results=True,output_dir=path)

	break

