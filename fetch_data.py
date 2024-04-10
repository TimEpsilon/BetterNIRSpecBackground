from astroquery.mast import Observations as OBS 
import numpy as np
import datetime
import time
from glob import glob
import os

def logConsole(text):
	curr_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
	print(f"[{curr_time}]  :  {text}")



logConsole("Starting MAST Query...")

obs_table = OBS.query_criteria(
	dataRights = ["public"],
	provenance_name = ["CALJWST"], 
	intentType = ["science"],
	obs_collection = ["JWST"],
	instrument_name = ["NIRSPEC/MSA"],
	obs_title = ['Spectroscopic follow-up of ultra-high-z candidates in CEERS: Characterizing true z > 12 galaxies and z~4-7 interlopers in preparation for JWST Cycle 2', 
					'The Cosmic Evolution Early Release Science (CEERS) Survey free'],
	filters = ["CLEAR;PRISM"]
	)

logConsole("Successful Query!")
print(obs_table)

n = 2
logConsole(f"Starting Product List Query for n={n}")
products_to_download = ['UNCAL', 'MSA', 'ASN']

for i in range(1,n) :
	logConsole("Product " + str(i))
	obsids = obs_table['obsid'][i]
	path = "./mastDownload/JWST/" + obs_table["obs_id"][i]

	if not os.path.exists(path):
		os.makedirs(path)

	data_products = OBS.get_product_list(obsids)

	mask = False
	for p_type in products_to_download:
		mask = np.logical_or(
			mask,
			data_products['productSubGroupDescription'] == p_type
			)

	data_filtered = data_products[mask]

	mask = False
	for i,name in enumerate(data_filtered["productFilename"]):
		if "image" in name:
			temp_obs_id = data_filtered[i]["obs_id"]
			mask = np.logical_or(
				mask,
				data_filtered["obs_id"] == temp_obs_id
				)

	data_filtered = data_filtered[np.logical_not(mask)]
	logConsole("Successful Filtering!")
	print(data_filtered)

	logConsole("Starting Download")
	OBS.download_products(data_filtered,flat=True,download_dir=path)