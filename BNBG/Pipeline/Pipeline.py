import os
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.CrdsSetup import getCRDSPath
from ..utils.logger import logConsole
from ..Pipeline import PipelineStages

# Needs to be overwritten in ../CRDS_PATH
os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from glob import glob


def main(directory,
		 folders,
		 skip1,
		 skip2,
		 skipbnbg,
		 skip3,
		 radius,
		 crop,
		 interpolationKnots,
		 curvatureConstraint,
		 endpointConstraint,
		 kernelSize,
		 Nsigma):
	"""
	Minimum requirement :
	An existing mastDownload/JWST folder containing multiple folders for multiple observations
	Each observation should contain at least an _uncal.fits files, a _msa.fits file and a _spec3 file
	Stage 2 should still take the _spec2 association files, as the multiple exposure are still used.
	args.directory,
	"""
	logConsole(f"Working directory is {directory}")

	if folders is None:
		folders = glob(os.path.join(directory,"*"))
	else:
		folders = [os.path.join(directory, f) for f in folders]

	for i, f in enumerate(folders):
		if not os.path.exists(f):
			logConsole(f"{f} does not exist!", source="WARNING")
			folders[i] = None

	folders = [f for f in folders if f is not None]

	logConsole(f"Found {len(folders)} folders : {folders}")


	##########
	# Stage 1
	##########
	for path in folders:
		logConsole(f"Starting on {os.path.basename(path)}")
		uncalList = glob(os.path.join(path, "*_uncal.fits"))
		logConsole(f"Found {len(uncalList)} uncalibrated files.")

		if not skip1:
			_runParallel(uncalList, lambda file : PipelineStages.Stage1(file, path))
		else:
			logConsole(f"Skipping Stage 1...")

	##########
	# Stage 2
	##########
	logConsole(f"Stage 1 Finished. Preparing Stage 2")

	for path in folders:
		logConsole(f"Starting on {os.path.basename(path)}")
		rateList = glob(os.path.join(path, "*_rate.fits"))
		logConsole(f"Found {len(rateList)} countrate files.")

		if not skip2:
			_runParallel(rateList, lambda file: PipelineStages.Stage2(file,
																	  path,
																	  radius=radius,
																	  crop=crop,
																	  interpolationKnots=interpolationKnots,
																	  curvatureConstraint=curvatureConstraint,
																	  endpointConstraint=endpointConstraint,
																	  kernelSize=kernelSize,
																	  Nsigma=Nsigma,
																	  skipbnbg=skipbnbg))
		else:
			logConsole(f"Skipping Stage 2...")

	##########
	# Stage 3
	##########
	logConsole(f"Stage 2 Finished. Preparing Stage 3")

	for path in folders:
		logConsole(f"Starting on {os.path.basename(path)}")
		asn = glob(os.path.join(path, "*_spec3_*_asn.json"))[0] # Only keep 1st _spec3
		logConsole(f"Found a _spec3 association files")

		if not skip3:
			PipelineStages.Stage3(asn, path)
		else:
			logConsole(f"Skipping Stage 3...")
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
				logConsole(traceback.format_exc())

if __name__ == "__main__":
	# $ python -m BNBG.Pipeline.Pipeline -<args>
	parser = ArgumentParser()

	parser.add_argument(
		"directory",
		type=str,
		help="Path to the main directory containing the subfolders for each observation. e.g. mastDownload/JWST/*")

	parser.add_argument(
		"-f", "--folders",
		type=str,
		nargs='+',
		default=None,
		help="List of folder paths relative to the main directory. Keep empty to select all found folders.")

	parser.add_argument(
		"--skip1",
		action="store_true",
		help="Skip 1st Stage (uncal -> rate)"
	)

	parser.add_argument(
		"--skip2",
		action="store_true",
		help="Skip 2nd Stage (rate -> cal/s2d)"
	)

	parser.add_argument(
		"--skipbnbg",
		action="store_true",
		help="Skip BNBG substage (cal/s2d -> bkg-BNBG/cal-BNBG)"
	)

	parser.add_argument(
		"--skip3",
		action="store_true",
		help="Skip 3rd Stage (cal-BNBG -> s2d/x1d)"
	)

	parser.add_argument(
		"--radius",
		type=float,
		help="Radius of extraction around source.",
		default=4
	)

	parser.add_argument(
		"--crop",
		type=int,
		help="Number of pixel lines to remove above and below the s2d image.",
		default=3
	)

	parser.add_argument(
		"--iknots",
		type=float,
		help="Interpolation Knots, "
			 "Fraction of total data points used as knots for the spline fitting (between 0 and 1).",
		default=0.15
	)

	parser.add_argument(
		"--curv",
		type=float,
		help="curvatureConstraint, "
			 "controls the smoothing of the spline. "
			 "Useful when gaps are present in the data. If set to 0, curvature is ignored.",
		default=1
	)

	parser.add_argument(
		"--endpoint",
		type=float,
		help="endpointConstraint, "
			 "controls the smoothing at the endpoints of the spline. "
			 "If set to 0, endpoint slopes are ignored.",
		default=0.1
	)

	parser.add_argument(
		"--kernel",
		type=int,
		nargs=2,
		help="Size of the kernel for median filtering. Needs to be exactly 2 integers",
		default=(1,15)
	)

	parser.add_argument(
		"-N", "--Nsigma",
		type=float,
		help="Number of sigmas around the median image for which to filter extreme pixels. "
			 "Here, sigma means median absolute deviation.",
		default=10
	)

	args = parser.parse_args()

	args.kernel = (args.kernel[0], args.kernel[1])

	main(args.directory,
		 args.folders,
		 args.skip1,
		 args.skip2,
		 args.skipbnbg,
		 args.skip3,
		 args.radius,
		 args.crop,
		 args.iknots,
		 args.curv,
		 args.endpoint,
		 args.kernel,
		 args.Nsigma)