import os
from glob import glob
from astropy.io import fits
from astropy.table import QTable, vstack
import numpy as np


working_dir = "../mastDownload/JWST/"

def iterateOverFolders():
	tables = []
	for folder in glob(working_dir + "*"):
		folder = f"{folder}/Final/"
		print(f"Working on {folder}")
		_ = generateCigaleFile(folder)
		if _ is None:
			continue
		tables.append(_)
	table = vstack(tables)

	table.write(working_dir + 'cigale-data.fits', overwrite=True, format="ascii")


def generateCigaleFile(folder):
	ids = []
	paths = []
	modes = []

	for file in glob(os.path.join(folder, '*_x1d.fits')):
		data = fits.open(file)

		# Excluding entirely nan arrays
		if not np.any(np.logical_and(np.isfinite(data[1].data['FLUX']), np.isfinite(data[1].data['FLUX_ERROR']))):
			continue

		program = folder.split("/")[-3].split("-")[-3]
		srcid = file.split("/")[-1].split("_")[-4]
		if "prism1" in file:
			srcid = srcid + "-1"
		ids.append(f"nirspec_{program}_{srcid}")
		paths.append(file)
		modes.append("prism")
		#modes.append(data[0].header["GRATING"].lower())

	if len(ids) == 0:
		print("Empty folder")
		return
	redshift = [-1 for _ in range(len(ids))]
	norm = ["wave" for _ in range(len(ids))]

	table = QTable([ids, redshift, paths, modes, norm],
				   names=('id', 'redshift', 'spectrum', 'mode', 'norm'))

	print("Saving Table")
	print(table)
	return table



iterateOverFolders()