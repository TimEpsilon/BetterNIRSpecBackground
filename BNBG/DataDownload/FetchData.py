#  Copyright (c) 2025. Tim Dewachter, LAM

from astroquery.mast import Observations as OBS
import numpy as np
import os
from astropy import table
from glob import glob
from astropy.io import fits


def download(obsids,path):
	if not os.path.exists(path):
		os.makedirs(path)

	data_products = OBS.get_product_list(obsids)

	# 1st filtering, only keeping interesting files
	mask = False
	for p_type in products_to_download:
		mask = np.logical_or(
			mask,
			data_products['productSubGroupDescription'] == p_type
			)

	# Removes unnecessary duplicates due to parent id
	data_filtered = table.unique(data_products[mask],silent=True,keys=["productFilename"])

	if len(data_filtered) == 0:
		pass

	print("Successful Filtering!")
	print(data_filtered)
	print("Starting Download")
	OBS.download_products(data_filtered
						  ,flat=True,download_dir=path)

	cleanup(path)


def cleanup(path):
	for f in glob(path+"*_uncal.fits"):
		with fits.open(f) as hdul:
			try:
				if hdul[0].header["EXP_TYPE"] != "NRS_MSASPEC":
					print("Deleting {}".format(f))
					os.remove(f)
			except :
				continue

	for f in glob(path+"*image2*.json"):
		os.remove(f)

########
# Main
########

print("Starting MAST Query...")

# List of interesting products :
# CAL : full image after stage 2
# RATE : full image after stage 1
# S2D : cutout after stage 2
# UNCAL : raw images
# MSA : used for calibration
# ASN : association files
products_to_download = ['MSA', 'ASN', 'UNCAL']
programs_to_ignore = ["CEERS-NIRSPEC-P10-MR-MSATA",
					"CEERS-NIRSPEC-P10-PRISM-MSATA",
					"CEERS-NIRSPEC-P11-PRISM-MSATA",
					"CEERS-NIRSPEC-P12-PRISM-MSATA",
					"CEERS-NIRSPEC-P4-MR-MSATA",
					"CEERS-NIRSPEC-P4-PRISM-MSATA",
					"CEERS-NIRSPEC-P5-MR-MSATA",
					"CEERS-NIRSPEC-P7-MR-MSATA",
					"CEERS-NIRSPEC-P7-PRISM-MSATA",
					"CEERS-NIRSPEC-P8-MR-MSATA",
					"CEERS-NIRSPEC-P8-PRISM-MSATA",
					"CEERS-NIRSPEC-P9-MR-MSATA",
					"CEERS-NIRSPEC-P9-PRISM-MSATA"]

obs_table = OBS.query_criteria(
	dataRights = ["public"],
	provenance_name = ["CALJWST"], 
	intentType = ["science"],
	obs_collection = ["JWST"],
	instrument_name = ["NIRSPEC/MSA"],
	obs_title = ['Spectroscopic follow-up of ultra-high-z candidates in CEERS: Characterizing true z > 12 galaxies and z~4-7 interlopers in preparation for BNBG Cycle 2',
					'The Cosmic Evolution Early Release Science (CEERS) Survey free'],
	dataproduct_type = ["spectrum"]
	)

print("Successful Query!")

for program in np.unique(obs_table["target_name"]):
	if program in programs_to_ignore:
		continue
	_ = obs_table[obs_table["target_name"] == program]
	print(f"Querying program: {program}")
	path = "../../mastDownload/JWST/" + program + "/"
	download(_["obsid"], path)

