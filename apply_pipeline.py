import os

os.environ['CRDS_PATH'] = '/home/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline
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
import sys
import pandas as pd

from utils import logConsole, numberSameLength, rewriteJSON


working_dir = "./mastDownload/JWST/"
folders = os.listdir(working_dir) #Default, needs to be overwritten
try :
	folders = sys.argv[1:]
except :
	logConsole("No Folders Specified. Defaulting to all Folders")

#########
# Monkey Patch Master Background method
#########

#BkgMosStep.process = BetterMasterBackgroundMosStep
#logConsole("Overriding Master Background method")


logConsole(f"Found {len(folders)} folders")
for folder in folders:
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

for folder in folders:
	path = working_dir + folder + "/"
	logConsole(f"Starting on {folder}")

	rate_list = glob(path+"*_rate.fits")
	logConsole(f"Found {len(rate_list)} countrate files")

	for n,rate in enumerate(rate_list):
		logConsole(f"Starting Stage 2 ({n+1}/{len(rate_list)})")
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

		if not os.path.exists(bkg.replace("_bkg","_bkg_photomstep")):
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
				calibrated = WavecorrStep.call(data)
				calibrated = FlatFieldStep.call(calibrated)
				calibrated = PathLossStep.call(calibrated)
				calibrated = BarShadowStep.call(calibrated)
				calibrated = PhotomStep.call(calibrated,output_dir=path,save_results=True)
				calibrated = PixelReplaceStep.call(calibrated)
				calibrated = ResampleSpecStep.call(calibrated,output_dir=path,save_results=True)
				calibrated = Extract1dStep.call(calibrated,output_dir=path,save_results=True)


	##########
	# Basic Master Background Algorithm
	# 0 - Apply Spec2 up until srctype (included) and save them
	# 1 - Modify the _srctype files, all spectra
	# 2 - Create a _bkg file, exact copy
	# 3 - Apply the rest of Spec2 to _bkg
	# 4 - Modify the spec3_asn.json
	# 5 - Apply Spec3 to the spec3_asn.json
	##########

	##########
	# Stage 3
	##########
	logConsole(f"Stage 2 Finished. Preparing Stage 3")

for folder in folders:
	path = working_dir + folder + "/"

	if os.path.exists(f"{path}FilesOfInterest.csv"):
		logConsole("FilesOfInterest.csv already exists. Skipping this folder")
		continue
	else:
		logConsole(f"Starting on {folder}")

		asn_list = glob(path+"*_spec3_*_asn.json")
		logConsole(f"Found {len(asn_list)} association files")


		for n,asn in enumerate(asn_list):
			logConsole(f"Starting Stage 3 ({n+1}/{len(asn_list)})")
			logConsole("Modifying Stage 3 association files")
			rewriteJSON(asn)

			final = path + "Final/"
			if not os.path.exists(final):
				os.makedirs(final)
			Spec3Pipeline.call(asn,save_results=True,output_dir=final)

		# Creates a file signifying that the pipeline has finished
		# As a bonus, this file acts as a table containing the names of the files mentioned in slits_with_double_object.dat
		double_slits = pd.read_csv("slits_with_double_object.dat", sep=",")
		main_target = double_slits["Central_target"]
		companion = double_slits["Companion"]
		main_target = main_target.apply(numberSameLength)
		companion = companion.apply(numberSameLength)

		target_path = []
		n = len(main_target)
		for i in range(n):
			target = main_target[i]
			_ = glob(f"{path}*{target}*_s2d.fits")
			if len(_) > 0 and f"P{double_slits['Pointing'][i]}" in _ [0]:
				target_path.append(_[0].split("/")[-1])
			else:
				target_path.append(None)

		for i in range(n):
			target = companion[i]
			_ = glob(f"{path}*{target}*_s2d.fits")
			if len(_) > 0 and f"P{double_slits['Pointing'][i]}" in _ [0]:
				target_path.append(_[0].split("/")[-1])
			else:
				target_path.append(None)

		file_of_interest = {"TargetType" : ["Main"]*len(main_target) + ["Companion"]*len(main_target),
							"ID" : [*main_target, *companion],
							"Path" : target_path}

		df = pd.DataFrame(file_of_interest)
		df.to_csv(f"{path}FilesOfInterest.csv")

