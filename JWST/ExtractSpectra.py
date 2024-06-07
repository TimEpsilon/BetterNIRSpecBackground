import os
os.environ['CRDS_PATH'] = '/net/CLUSTER/VAULT/users/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import numpy as np
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import logConsole
from astropy.visualization import ZScaleInterval
from multiprocessing import Pool, cpu_count
from jwst.extract_1d import Extract1dStep as x1d
from astropy.io import fits


working_dir = "./mastDownload/JWST/"
folders = glob(working_dir+'*')
num_processes = min(len(folders), cpu_count())

def IterateOverFolders(folders):
	pool_obj = Pool(num_processes)
	pool_obj.map(WorkOn1Folder,folders)
	pool_obj.close()

def WorkOn1Folder(folder):
	path = folder + "/Final/"
	if os.path.exists(path):
		logConsole(f"Working on {folder}")
		for file in glob(path + "*_s2d.fits"):
			logConsole(f"Working on {file}")
			makeExtraction(file)


def makeExtraction(file):
	"""
	Extracts a spectrum from a _s2d file and saves it as a .png and a fits
	Parameters
	----------
	file : str
		The path to the _s2d file
	Returns
	-------

	"""
	# Skip if not 3 shutter slit
	with dm.open(file) as mos:
		if len(mos["shutter_state"]) != 3:
			return

		# Slice height
		n = 2

		# Apply extract 1D
		step = x1d()
		step.smoothing_length = 2 * n + 1
		step.use_source_posn = True
		step.save_results = True
		step.suffix = "x1d"
		step.output_dir = os.path.join(*file.split("/")[:-1])
		step(mos)

		# Get from extraction
		with fits.open(file.replace("_s2d.fits", "_x1d.fits")) as hdul:
			wave = hdul[1].data["WAVELENGTH"]
			flux = hdul[1].data["FLUX"]
			dflux = hdul[1].data["FLUX_ERROR"]
			ymin, ymax = hdul[1].header["EXTRYSTR"], hdul[1].header["EXTRYSTP"]

		# Convert data to Jy
		data = mos.data
		if mos.meta.bunit_data == "MJy":
			unit = "Jy"
			_ = 1e6
		elif mos.meta.bunit_data == "MJy/sr":
			unit = "Jy"
			_ = mos.meta.photometry.pixelarea_steradians * 1e6
		else:
			unit = mos['meta'].bunit_data
			_ = 1
		data *= _


		# Z scale algorithm
		z = ZScaleInterval()
		z1, z2 = z.get_limits(data)

		# The 2D plot
		fig, ax = plt.subplots(2, figsize=(12, 10), gridspec_kw={'hspace': 0})
		fig.subplots_adjust(hspace=0)
		im = ax[0].imshow(data, interpolation='None', origin="lower", vmin=z1, vmax=z2)
		divider = make_axes_locatable(ax[0])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		fig.colorbar(im, cax=cax, orientation='vertical', label=f"Flux ({unit})")

		# Visual slice
		ax[0].hlines((ymin, ymax), 0, data.shape[1], color='r', linewidth=0.5, linestyle='dashed')
		ax[0].xaxis.set_ticks_position('top')
		ax[0].xaxis.set_label_position('top')
		ax[0].margins(x=0)

		# The spectrum plot
		ax[1].plot(wave, flux, marker='+', color="k", linewidth=0.5)
		ax[1].fill_between(wave, flux + dflux, flux - dflux, color='gray', alpha=0.5)
		ax[1].grid(True)
		ax[1].set_xlabel(r"$\lambda$ measured (µm)", fontsize=14)
		ax[1].set_ylabel(fr"$F_\lambda$ ({unit})", fontsize=14)
		ax[1].margins(x=0)

		# bounding box
		box = ax[1].get_position()
		box.y0 += 0.137
		box.y1 += 0.137
		ax[1].set_position(box)

		name = file.replace("_s2d.fits", "_extracted.png")
		logConsole(f"Saving {name}")
		plt.savefig(name)

IterateOverFolders(folders)