import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from BNBG.Pipeline import MainPipeline
from BNBG.utils import getCRDSPath, logConsole

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
	script_dir = os.path.dirname(os.path.dirname(script_dir))
	working_dir = os.path.join(script_dir, 'mastDownload/JWST/')

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

		n = min(os.cpu_count(), len(rate_list))
		# Execute in parallel
		with ThreadPoolExecutor(max_workers=n) as executor:
			futures = [executor.submit(lambda file : MainPipeline.Stage2(file, path), file) for file in rate_list]

			# Wait for all futures to complete
			for future in as_completed(futures):
				try:
					future.result()
				except Exception as e:
					logConsole(f"Error processing a file: {e}")

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

	############
	# Stage 4
	# Custom Subtraction
	############
	logConsole(f"Stage 3 Finished. Preparing Final Stage")

	for folder in folders:
		path = working_dir + folder + "/Final/"

		logConsole(f"Starting on {folder}")
		s2d_list = glob(path+"*_s2d.fits")
		logConsole(f"Found {len(s2d_list)} s2d files.")

		path = path + "BNBG/"
		if not os.path.exists(path):
			os.mkdir(path)

		for s2d in s2d_list:
			MainPipeline.Stage4(s2d, path)

if __name__ == "__main__":
	main()