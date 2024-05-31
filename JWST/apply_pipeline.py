import os

from jwst.pipeline import Spec3Pipeline

import MainPipeline

os.environ['CRDS_PATH'] = '/home/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from multiprocessing import Pool, cpu_count

from glob import glob

import sys
import pandas as pd

from utils import logConsole, numberSameLength, rewriteJSON


def main():
	working_dir = "./mastDownload/JWST/"
	folders = os.listdir(working_dir) #Default, needs to be overwritten
	try :
		folders = sys.argv[1:]
	except :
		logConsole("No Folders Specified. Defaulting to all Folders")


	logConsole(f"Found {len(folders)} folders")
	for folder in folders:
		path = working_dir + folder + "/"
		logConsole(f"Starting on {folder}")

		##########
		# Stage 1
		##########


		uncal_list = glob(path+"*_uncal.fits")
		num_processes = min(len(uncal_list), cpu_count())
		logConsole(f"Found {len(uncal_list)} uncalibrated files. Running on {num_processes} threads")

		args = [(file, path) for file in uncal_list]

		# Open threads
		pool_obj = Pool(num_processes)
		pool_obj.starmap(MainPipeline.Stage1, args)
		pool_obj.close()


	##########
	# Stage 2
	##########

	logConsole(f"Stage 1 Finished. Preparing Stage 2")

	for folder in folders:
		path = working_dir + folder + "/"
		logConsole(f"Starting on {folder}")

		rate_list = glob(path+"*_rate.fits")
		num_processes = min(len(rate_list), cpu_count())
		logConsole(f"Found {len(rate_list)} countrate files. Running on {num_processes} threads")

		args = [(file, path) for file in rate_list]

		# Open threads
		pool_obj = Pool(num_processes)
		pool_obj.starmap(MainPipeline.Stage2, args)
		pool_obj.close()


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
			num_processes = min(len(asn_list), cpu_count())
			logConsole(f"Found {len(asn_list)} association files. Running on {num_processes} threads")

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

			logConsole("Finished")

if __name__ == "__main__":
	main()