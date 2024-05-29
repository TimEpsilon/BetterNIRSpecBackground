import os
from glob import glob
from astropy.io import fits
from astropy.table import QTable
from utils import logConsole


working_dir = "./mastDownload/JWST/"

def iterateOverFolders():
	for folder in glob(working_dir + "*"):
		folder = f"{folder}/Final/"
		print(f"Working on {folder}")
		generateCigaleFile(folder)


def generateCigaleFile(folder):
	ids = []
	paths = []
	modes = []

	for file in glob(os.path.join(folder, '*_x1d.fits')):
		data = fits.open(file)

		# Ignoring none 3 shutter slits because they have not been treated
		if len(data[1].header["SHUTSTA"]) != 3:
			continue

		ids.append(f"nirspec_{data[1].header['SOURCEID']}")
		paths.append(file)
		modes.append(data[0].header["GRATING"].lower())

	if len(ids) == 0:
		print("Empty folder")
		pass
	redshift = [-1 for _ in range(len(ids))]
	norm = ["wave" for _ in range(len(ids))]

	table = QTable([ids, redshift, paths, modes, norm],
				   names=('id', 'redshift', 'spectrum', 'mode', 'norm'))

	print("Saving Table")
	print(table)
	table.write(folder + 'fin', overwrite=True, format="ascii")


iterateOverFolders()