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
	Each observation should contain at least an _uncal.fits files, a _msa.fits file and a _spec3 file
	Stage 2 should still take the _spec2 association files, as the multiple exposure are still used.
	"""

	thisDirectory = os.path.dirname(os.path.abspath(__file__))
	thisDirectory = os.path.dirname(os.path.dirname(thisDirectory)) # Go up 2 directories
	workingDirectory = os.path.join(thisDirectory, 'mastDownload/JWST/')

	folders = os.listdir(workingDirectory) #Default, needs to be overwritten
	try :
		# If file is executed from command prompt, get folders passed as argument
		# $ python ApplyPipeline.py dir1 dir2 dir3 ...
		_ = sys.argv[1:]
		if len(_) > 0:
			folders = [f for f in folders if f in _]
			logConsole(f"Found {len(folders)} folders")
	except IndexError:
		logConsole("No Folders Specified. Defaulting to all Folders", source="WARNING")


	##########
	# Stage 1
	##########
	for folder in folders:
		logConsole(f"Starting on {folder}")
		path = os.path.join(workingDirectory, folder)
		uncalList = glob(os.path.join(path, "*_uncal.fits"))
		logConsole(f"Found {len(uncalList)} uncalibrated files.")

		_runParallel(uncalList, lambda file : MainPipeline.Stage1(file, path))

	##########
	# Stage 2
	##########
	logConsole(f"Stage 1 Finished. Preparing Stage 2")

	for folder in folders:
		logConsole(f"Starting on {folder}")
		path = os.path.join(workingDirectory, folder)
		rateList = glob(os.path.join(path, "*_rate.fits"))
		logConsole(f"Found {len(rateList)} countrate files.")

		_runParallel(rateList, lambda file: MainPipeline.Stage2(file, path))

	##########
	# Stage 3
	##########
	logConsole(f"Stage 2 Finished. Preparing Stage 3")

	for folder in folders:
		logConsole(f"Starting on {folder}")
		path = os.path.join(workingDirectory, folder)
		asn = glob(path + "*_spec3_*_asn.json")[0] # Only keep 1st _spec3
		logConsole(f"Found a _spec3 association files")

		MainPipeline.Stage3(asn, path)

	logConsole("Finished")

def _runParallel(files : list, function : callable):
	"""
	Replaces a for loop but runs in parallel. Probably not threadsafe, whatever that means.

	Parameters
	----------
	files : list
		List of file paths

	function : callable
		Function to apply to each file. Takes a single file as parameter
	"""
	n = min(os.cpu_count(), len(files))

	# Execute in parallel
	with ThreadPoolExecutor(max_workers=n) as executor:
		futures = [executor.submit(lambda file: function(file), file) for file in files]

		# Wait for all futures to complete
		for future in as_completed(futures):
			try:
				future.result()
			except Exception as e:
				logConsole(f"Error processing a file: {e}")

if __name__ == "__main__":
	main()