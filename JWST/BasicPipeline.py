import os

os.environ['CRDS_PATH'] = '/home/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline

from glob import glob
import pandas as pd
import time

from utils import logConsole, numberSameLength
import json

start = time.time()

path = "./mastDownload/JWST/CEERS-NIRSPEC-P10-PRISM-MSATA/basicPipeline/"


########
# This is the full jwst pipeline with no modifications.
# This is a test file
########


def rewriteJSON(file):

	with open(file, "r") as asn:
		data = json.load(asn)

		# Get calibration indices
		not_science = []
		for i in range(len(data["products"][0]["members"])):
			if not data["products"][0]["members"][i]["exptype"] == "science":
				not_science.append(i)

		# Starting from the end in order to keep the values at the same index
		for i in not_science[::-1]:
			del data["products"][0]["members"][i]

	with open(file, "w") as asn:
		json.dump(data, asn, indent=4)


##########
# Stage 1
##########


uncal_list = glob(path + "*_uncal.fits")

for uncal in uncal_list:
	logConsole(f"Starting Stage 1 on {uncal}")

	if not os.path.exists(uncal.replace("_uncal", "_rate")):
		Detector1Pipeline.call(uncal,save_results=True,output_dir=path)

##########
# Stage 2
##########

logConsole(f"Stage 1 Finished. Preparing Stage 2")

rate_list = glob(path+"*_rate.fits")
for rate in rate_list:
	logConsole(f"Starting Stage 2 on {rate}")

	if not os.path.exists(rate.replace("rate","cal")):
			Spec2Pipeline.call(rate,save_results=True,output_dir=path)

##########
# Stage 3
##########
logConsole(f"Stage 2 Finished. Preparing Stage 3")

if os.path.exists(f"{path}FilesOfInterest.csv"):
	logConsole("FilesOfInterest.csv already exists. Skipping this folder")
else:
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

	logConsole(f"Finished with time {round(time.time() - start)}s")



