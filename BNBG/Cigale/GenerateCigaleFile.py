import os
from glob import glob
from astropy.io import fits
from astropy.table import QTable, vstack
from argparse import ArgumentParser
import pandas as pd

def main(directory, folders, redshifts, suffix):
	"""
	Allows to generate a CIGALE input file ::

		$ python GenerateCigaleFile.py --directory <directory> --folders <folders> --redshifts <redshifts>

	Parameters
	----------
	directory : str
		Working directory which will be used as the starting point for each following relative path

	folders : list of str
		The relative paths to each folder containing the files to add to the table

	redshifts : list of str
		The relative paths to each csv containing the table mapping each source to its redshift. Can be None

	suffix : list of str
		suffixes to search within the folders
	"""
	folders = [os.path.join(directory,folder) for folder in folders]
	redshifts = [pd.read_csv(os.path.join(directory, redshift)) if not redshift is None else None for redshift in redshifts]
	n = 0
	table1 = None
	for i in range(len(folders)):
		for j,s in enumerate(suffix):
			folder = folders[i]
			redshift = redshifts[i]
			table2 = generateCigaleFile(folder, suffix=s, redshift_map=redshift)
			if table1 is None:
				table1 = table2
			else:
				table1 = combineTable(table1, table2, n)
			n+=1
	table1.sort("id")
	print(table1)
	table1.write(os.path.join(directory, 'cigale-data.fits'), overwrite=True, format='ascii')

def combineTable(table1, table2, n):
	"""
	Combines 2 tables into 1. The ids of table2 will be appended with a _n

	Parameters
	----------
	table1 : Table
		First table, its values will remain unchanged.

	table2 : Table
		Second table, its values will be changed to id_n.

	n : int
		Discriminate between same sources.

	Returns
	-------
	result : Table
		Table containing the combined data.

	"""
	table1["id"] = table1["id"].astype("str")
	table2["id"] = table2["id"].astype("str")
	table2["id"] = [f"{str(src).split('_')[0]}_{n}" for src in table2["id"]]
	return vstack([table1, table2])


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
	table : astropy.table.QTable
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
				redshift.append(match["best.universe.redshift"].iloc[0])

	norm = ["wave" for _ in range(len(ids))]

	table = QTable([ids, redshift, paths, modes, norm],
				   names=('id', 'redshift', 'spectrum', 'mode', 'norm'))
	return table


#####################
# 		MAIN
#####################

if __name__ == "__main__":
	# $ python GenerateCigaleFile.py --args
	parser = ArgumentParser()

	parser.add_argument(
		"--directory",
		type=str,
		required=True,
		help="Path to the main directory")

	parser.add_argument(
		"--folders",
		type=str,
		nargs='+',
		help="List of folder paths relative to the main directory")

	parser.add_argument(
		"--redshifts",
		type=str,
		nargs='+',
		help="Relative paths to redshift CSV files (must match the number of folders, use 'None' for empty entries)"
	)

	parser.add_argument(
		"--suffix",
		type=str,
		nargs='+',
		help="Suffixes to search within the files, such as 'x1d' or 'x1d_optext'"
	)

	args = parser.parse_args()

	if len(args.folders) != len(args.redshifts):
		parser.error("The number of redshifts must match the number of folders")

	args.redshifts = [None if r.lower() == 'none' else r for r in args.redshifts]

	main(args.directory, args.folders, args.redshifts, args.suffix)