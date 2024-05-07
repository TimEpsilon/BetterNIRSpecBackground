import numpy as np
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from utils import logConsole
from astropy.visualization import ZScaleInterval


working_dir = "./mastDownload/JWST/"
folders = glob(working_dir+'*')

def IterateOverFolders(folders):
	for folder in folders:
		path = folder + "/Final/"
		if os.path.exists(path):
			logConsole(f"Working on {folder}")
			for file in glob(path + "*_s2d.fits"):
				logConsole(f"Working on {file}")
				makeExtraction(file)



def makeExtraction(file):
	"""
	Extracts a spectrum from a _s2d file and saves it as a .png
	Parameters
	----------
	file : str
		The path to the _s2d file

	Returns
	-------

	"""
	with dm.open(file) as mos:
		if len(mos["shutter_state"]) != 3:
			pass

		# The position of the source
		# The slice will be made around this
		s_dec, s_ra = mos["source_dec"], mos["source_ra"]

		# Map of WCS
		Y, X = np.indices(mos["data"].shape)
		ra, dec, lamb = mos["meta"].wcs.transform('detector', 'world', X, Y)

		# Z scale algorithm
		z = ZScaleInterval()
		z1, z2 = z.get_limits(mos["data"])

		# The 2D plot
		fig, ax = plt.subplots(2, figsize=(10, 6), tight_layout=True)
		im = ax[0].imshow(mos["data"], interpolation='None', origin="lower", vmin=z1, vmax=z2)
		divider = make_axes_locatable(ax[0])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical', label=f"Flux ({mos['meta'].bunit_data})")

		# The slice
		n = 3
		x, y = mos["meta"].wcs.transform("world", "detector", s_ra, s_dec, lamb[0][0])
		ax[0].hlines((y + n, y - n), 0, Y.shape[1], color='r', linewidth=0.5, linestyle='dashed')

		wavelength = lamb[round(y) - n:round(y) + n, :]
		target = mos["data"][round(y) - n:round(y) + n, :]
		err = mos["err"][round(y) - n:round(y) + n, :]


		wavelength = wavelength.mean(axis=0)
		spectrum = target.mean(axis=0)
		# Usually, the error is dominated by the variance on the slice
		err = np.mean(err ** 2, axis=0) + target.var(axis=0)
		err = np.sqrt(err)

		# The spectrum plot
		ax[1].errorbar(wavelength, spectrum, yerr=err, marker='.', color="k", linewidth=0.3)
		ax[1].grid(True)

		name = file.split("/")[-1].replace("_s2d.fits", ".png")
		logConsole(f"Saving {name}")
		plt.savefig(file.replace("_s2d.fits", ".png"))
