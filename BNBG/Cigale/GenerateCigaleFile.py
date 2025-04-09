import os
from glob import glob
from astropy.io import fits
from astropy.table import Table, vstack, join
from argparse import ArgumentParser
import pandas as pd

def main(directory, folders, redshifts, photom, force=False):
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

	photom : list of str
		The relative paths to each file containing the table mapping each source to its photometry. Can be None

	force : bool
		Whether the tables should be forced combined.
	"""
	folders = [os.path.join(directory,folder) for folder in folders]
	redshifts = [pd.read_csv(os.path.join(directory, redshift)) if not redshift is None else None for redshift in redshifts]
	photom = [Table.read(os.path.join(directory, _)) if not _ is None else None for _ in photom]
	table1 = None
	for i in range(len(folders)):
		folder = folders[i]
		redshift = redshifts[i]
		pho = photom[i]
		table2 = generateCigaleFile(directory, folder, redshift_map=redshift, photom=pho)
		if table1 is None:
			table1 = table2
		else:
			table1 = combineTable(table1, table2, i, force=force)

	table1.sort("id")
	print(table1)
	table1.write(os.path.join(directory, 'cigale-data.fits'), overwrite=True, format='fits')

def combineTable(table1, table2, n, force=False):
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

	force : bool
		If True, the table1 will be kept and only the ids in table2 which aren't in table1 will be added.

	Returns
	-------
	result : Table
		Table containing the combined data.

	"""
	table1["id"] = table1["id"].astype("str")
	table2["id"] = table2["id"].astype("str")
	if not force:
		table2["id"] = [f"{str(src).split('_')[0]}_{n}" for src in table2["id"]]
		return vstack([table1, table2])
	else:
		table = vstack([table1, table2])
		table = table.group_by("id")
		table = table[table.groups.indices[:-1]]
		return table



def generateCigaleFile(directory, folder, redshift_map=None, photom=None):
	"""
	Get a table formated for CIGALE with data from a folder.

	Parameters
	----------
	directory : str
		Working directory which will be used as the starting point for each following relative path

	folder : str
		Absolute path to folder containing _suffix files

	redshift_map : pd.DataFrame, optional
		A dataframe with a column "id" and another column "redshift", allowing to map a redshift to each object.
		If None, will default to -1

	photom : Table, optional
		A table with a column 'id' and multiple columns 'filter' and 'filter_err'.

	Returns
	-------
	table : astropy.table.Table
		A table formatted as such : id, redshift, spectrum, mode, norm, and the filters of photom
	"""
	ids = []
	paths = []
	modes = []

	for file in glob(os.path.join(folder, f'*_x1d.fits')):
		data = fits.open(file)
		file = os.path.relpath(file, directory)
		file = str(file).encode('utf-8', 'ignore') # Encode in utf-8

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

	# Photometry mapping
	if photom is not None:
		table = join(table, photom, keys="id", join_type="left")

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
		"--photom",
		type=str,
		nargs='+',
		help="Relative paths to photometry tables to use"
	)

	parser.add_argument(
		"--force",
		action="store_true",
		help="Force combining of tables",
		default=False
	)

	args = parser.parse_args()

	if len(args.folders) != len(args.redshifts):
		parser.error("The number of redshifts must match the number of folders")

	args.redshifts = [None if r.lower() == 'none' else r for r in args.redshifts]

	if len(args.folders) != len(args.photom):
		parser.error("The number of photometry must match the number of folders")

	args.photom = [None if r.lower() == 'none' else r for r in args.photom]

	main(args.directory, args.folders, args.redshifts, args.photom, args.force)