#  Copyright (c) 2025. Tim Dewachter, LAM

import os

def getCRDSPath() -> str:
	"""
	Get the path to the CRDS file from "./CRDS_PATH.txt"

	Returns
	-------
	txt : str
		Absolute path to the CRDS file

	"""
	# Get the directory of the current script
	scriptDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	path = os.path.join(scriptDirectory, 'CRDS_PATH.txt')
	with open(path) as file:
		txt = file.readline()
		if txt is None or txt == "":
			raise Exception("CRDS_PATH.txt not found or file empty")
		print(f"CRDS folder at {txt}")
		return txt