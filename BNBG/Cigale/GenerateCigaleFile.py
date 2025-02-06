import os
from glob import glob
from astropy.io import fits
from astropy.table import Table, vstack
import sys

def iterateOverFolders(working_dir, args):
	tables = []
	for directory in args:
		directory = os.path.join(working_dir, directory)
		if not os.path.exists(directory):
			print(f"{directory} does not exist")
			continue
		_ = generateCigaleFile(directory)
		tables.append(_)
	table = vstack(tables)

	table.write(working_dir + 'cigale-data.fits', overwrite=True, format="ascii")


def generateCigaleFile(folder, suffix="x1d", redshift_map=None):
	"""
	Get a table formated for CIGALE with data from a folder.

	Parameters
	----------
	folder : str
		Absolute path to folder containing _suffix files

	suffix : str, optional
		The suffix to look for in files

	redshift_map : pd.DataFrame, optional
		A dataframe with a column "id" and another column "redshift", allowing to map a redshift to each object.
		If None, will default to -1

	Returns
	-------
	table : astropy.table.Table
		A table formatted as such : id, redshift, spectrum, mode, norm
	"""
	ids = []
	paths = []
	modes = []

	for file in glob(os.path.join(folder, f'*_{suffix}.fits')):
		data = fits.open(file)

		# Getting file info
		srcid = data[1].header["SOURCEID"]
		grating = data[0].header["GRATING"].lower()
		ids.append(srcid)
		paths.append(file)
		modes.append(grating)

		data.close()

	# Redshift mapping
	if redshift_map is None:
		redshift = [-1 for _ in range(len(ids))]
	else :
		redshift = []
		for srcid in ids:
			match = redshift_map.loc[redshift_map["id"] == srcid]
			if match.empty:
				redshift.append(None)
			else:
				redshift.append(match["redshift"].iloc[0])

	norm = ["wave" for _ in range(len(ids))]

	table = Table([ids, redshift, paths, modes, norm],
				   names=('id', 'redshift', 'spectrum', 'mode', 'norm'))
	return table


#####################
# 		MAIN
#####################

if __name__ == "__main__":
	try:
		# 1st argument must be folder from which every other path will be relative to
		working_dir = sys.argv[1]
		if not os.path.exists(working_dir):
			raise Exception
		args = sys.argv[2:]
		iterateOverFolders(working_dir, args)

	except IndexError:
		print("No Folders Specified")