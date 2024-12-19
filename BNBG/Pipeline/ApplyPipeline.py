import os

from BNBG.Pipeline import MainPipeline
from BNBG.utils import getCRDSPath, logConsole
import shutil

# Needs to be overwritten in ../CRDS_PATH
os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from glob import glob

import sys


def main():
	"""
	Minimum requirement :
	An existing mastDownload/JWST folder containing multiple folders for multiple observations
	Each observation should contain at least a _uncal.fits files, a _msa.fits file and a _spec3.json file
	If the default subtraction is to be applied, the corresponding _spec2.json files should also exist
	"""

	script_dir = os.path.dirname(os.path.abspath(__file__))
	working_dir = os.path.join(script_dir, '../../mastDownload/JWST/')

	folders = os.listdir(working_dir) #Default, needs to be overwritten
	try :
		_ = sys.argv[1:]
		if len(_) > 0:
			folders = [f for f in folders if f in _]
	except :
		logConsole("No Folders Specified. Defaulting to all Folders")
	logConsole(f"Found {len(folders)} folders")


	##########
	# Stage 1
	##########

	for folder in folders:
		logConsole(f"Starting on {folder}")
		path = working_dir + folder + "/"
		uncal_list = glob(path+"*_uncal.fits")
		logConsole(f"Found {len(uncal_list)} uncalibrated files.")

		for file in uncal_list:
			MainPipeline.Stage1(file, path)

	##########
	# Stage 2
	##########

	logConsole(f"Stage 1 Finished. Preparing Stage 2")

	for folder in folders:
		path = working_dir + folder + "/"
		logConsole(f"Starting on {folder}")
		rate_list = glob(path+"*_rate.fits")
		logConsole(f"Found {len(rate_list)} countrate files.")

		##########
		# Basic Master Background Algorithm
		# 0 - Apply Spec2 up until srctype (included) and save them
		# 1 - Modify the _srctype files, all spectra
		# 2 - Create a _BNBG file, exact copy
		# 3 - Apply the rest of Spec2 to _BNBG
		# 4 - Modify the spec3_asn.json
		# 5 - Apply Spec3 to the spec3_asn.json
		##########

		for file in rate_list:
			MainPipeline.Stage2(file, path)

		##########
		# Stage 3
		##########

	logConsole(f"Stage 2 Finished. Preparing Stage 3")

	for folder in folders:
		path = working_dir + folder + "/"

		logConsole(f"Starting on {folder}")
		asn_list = glob(path + "*_spec3_*_asn.json")
		logConsole(f"Found {len(asn_list)} association files")

		MainPipeline.Stage3_AssociationFile(asn_list, path)

		logConsole("Finished")

if __name__ == "__main__":
	main()