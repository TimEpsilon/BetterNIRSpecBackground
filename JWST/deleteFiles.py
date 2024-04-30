import os
from glob import glob

working_dir = "./mastDownload/JWST/"

programs = glob(working_dir+"*")

for P in programs:
	files = glob(P+"/*")
	for file in files:
		if "_uncal" in file:
			continue
		if ".json" in file:
			continue
		if "_msa" in file:
			continue
		if "_rate" in file:
			continue

		print(file)