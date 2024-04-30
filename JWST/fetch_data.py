from astroquery.mast import Observations as OBS 
import numpy as np
import os
import pandas as pd

from utils import numberSameLength


def find_corresponding_entry(entry):
	"""
	Gets the obsid and obs_id for a given target id
	Parameters
	----------
	entry : a central target id

	Returns
		obsid, obs_id
	-------

	"""
	entry = numberSameLength(entry)
	for _ in range(len(obs_id)):
		obs = obs_id[_]
		if f"{entry}_nirspec" in obs:
			return obs_table["obsid"][_],obs_table["target_name"][_]

def download(obsids,path):
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

	mask = np.full(len(data_filtered["obs_id"]),False)
	for i,name in enumerate(data_filtered["productFilename"]):
		if "image" in name:
			temp_obs_id = data_filtered[i]["obs_id"]
			mask = np.logical_or(
				mask,
				data_filtered["obs_id"] == temp_obs_id
				)

	data_filtered = data_filtered[np.logical_not(mask)]

	if len(data_filtered) == 0:
		pass

	print("Successful Filtering!")
	print(data_filtered)
	print("Starting Download")
	OBS.download_products(data_filtered,flat=True,download_dir=path)

########
# Main
########




print("Starting MAST Query...")

products_to_download = ['UNCAL', 'MSA', 'ASN']
obs_table = OBS.query_criteria(
	dataRights = ["public"],
	provenance_name = ["CALJWST"], 
	intentType = ["science"],
	obs_collection = ["JWST"],
	instrument_name = ["NIRSPEC/MSA"],
	obs_title = ['Spectroscopic follow-up of ultra-high-z candidates in CEERS: Characterizing true z > 12 galaxies and z~4-7 interlopers in preparation for JWST Cycle 2', 
					'The Cosmic Evolution Early Release Science (CEERS) Survey free']
	)

print("Successful Query!")

# Priority to known double objects

obs_id = obs_table["obs_id"]
double_slits = pd.read_csv("slits_with_double_object.dat",sep=",")
target = double_slits["Central_target"]

result = target.apply(find_corresponding_entry)

print(f"Starting Product List Query for n={len(result)} known double objects")

for i in range(len(result)):
	if result[i] is None:
		continue
	print("Product " + str(i + 1))
	path = "./mastDownload/JWST/" + result[i][1]
	obsids = result[i][0]

	download(obsids,path)


n = len(obs_table['obsid'])
print(f"Starting Product List Query for n={n}")

for i in range(n) :
	print("Product " + str(i+1))
	obsids = obs_table['obsid'][i]
	path = "./mastDownload/JWST/" + obs_table["target_name"][i]

	download(obsids,path)

