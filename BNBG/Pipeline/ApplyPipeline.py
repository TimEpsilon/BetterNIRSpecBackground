import os
from BNBG.utils import getCRDSPath

# Needs to be overwritten in ../CRDS_PATH
os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import MainPipeline
from glob import glob

import sys

from BNBG.utils import logConsole


def main():
	working_dir = "../DataDownload/mastDownload/JWST/"
	folders = os.listdir(working_dir) #Default, needs to be overwritten
	try :
		folders = sys.argv[1:]
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

		for file in rate_list:
			MainPipeline.Stage2(file, path)


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
			asn_list = glob(path + "*_spec3_*_asn.json")
			logConsole(f"Found {len(asn_list)} association files")

			MainPipeline.Stage3_AssociationFile(asn_list, path)
			MainPipeline.Stage3_FinishUp(path)

			logConsole("Finished")

if __name__ == "__main__":
	main()