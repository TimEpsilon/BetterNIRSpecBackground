import matplotlib
matplotlib.use('Qt5Agg')

import os
os.environ['CRDS_PATH'] = './crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import numpy as np
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import logConsole
from astropy.visualization import ZScaleInterval
from jwst.extract_1d import Extract1dStep as x1d
from astropy.io import fits

working_dir = "./mastDownload/JWST/"
folders = glob(working_dir+'*PRISM*')
done = ["CEERS-NIRSPEC-P11-PRISM-MSATA",
		"CEERS-NIRSPEC-P5-PRISM-MSATA",
		"CEERS-NIRSPEC-P4-PRISM-MSATA",
		"CEERS-NIRSPEC-P12-PRISM-MSATA",
		"CEERS-NIRSPEC-P10-PRISM-MSATA"]

def IterateOverFolders(folders):
	for folder in folders:
			if any(d in folder for d in done):
				continue
			WorkOn1Folder(folder)

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

		# 1st click : do we keep the main spectrum
		# 2nd click : where is the second spectrum
		clicks = []
		DoWeExtract = False

		def onclick(event):
			nonlocal clicks
			nonlocal DoWeExtract
			if event.button == 3:  # Right click
				DoWeExtract = True
				logConsole("Extracting main slit")
				logConsole("Closing")
				plt.close()
				return
			elif event.inaxes and event.button == 1: # Left Click
				clicks.append(event.ydata)
				logConsole(f"Extracting secondary slit at {clicks[-1]}")
				return

		fig, ax = plt.subplots(figsize=(12,6))
		z = ZScaleInterval()
		z1, z2 = z.get_limits(mos.data)
		ax.imshow(mos.data, vmin=z1, vmax=z2, origin='lower',interpolation="none")
		fig.canvas.mpl_connect('button_press_event', onclick)
		plt.show()

		print(clicks, DoWeExtract)

		if DoWeExtract:
			ExtractMainSource(file, n, mos)

		for i,yExtract in enumerate(clicks):
			if np.isnan(yExtract):
				break
			ExtractSecondarySource(file, n, mos, yExtract, i+1)

def ExtractMainSource(file, n, mos):
	# Extracts if not done already
	# The main slit
	if not os.path.exists(file.replace("s2d", "x1d")):
		# Apply extract 1D
		step = x1d()
		step.smoothing_length = 2 * n + 1
		step.use_source_posn = True
		step.save_results = True
		step.suffix = "x1d"
		step.output_dir = os.path.dirname(file)
		step(mos)

	# Get from extraction
	with fits.open(file.replace("_s2d.fits", "_x1d.fits")) as hdul:
		wave = hdul[1].data["WAVELENGTH"]
		flux = hdul[1].data["FLUX"]
		dflux = hdul[1].data["FLUX_ERROR"]
		ymin, ymax = hdul[1].header["EXTRYSTR"], hdul[1].header["EXTRYSTP"]
		unitx1d = hdul[1].header["TUNIT2"]

	# Convert data to Jy
	data = np.copy(mos.data)
	factor, unit = changeUnit(mos.meta.bunit_data,mos.meta.photometry.pixelarea_steradians)
	data *= factor

	# Z scale algorithm
	z = ZScaleInterval()
	z1, z2 = z.get_limits(data)

	# The 2D plot
	fig, ax = plt.subplots(2, figsize=(12, 10), gridspec_kw={'hspace': 0})
	fig.subplots_adjust(hspace=0)
	im = ax[0].imshow(data, interpolation='None', origin="lower", vmin=z1, vmax=z2)
	divider = make_axes_locatable(ax[0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im, cax=cax, orientation='vertical', label=f"Flux ({unitx1d})")

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
	plt.close()


def ExtractSecondarySource(file, n, mos, yExtract, extractId):
	yExtract = round(yExtract)
	data = np.copy(mos.data)
	# Convert data to Jy
	factor, unit = changeUnit(mos.meta.bunit_data,mos.meta.photometry.pixelarea_steradians)
	data *= factor

	# Extract from data
	Y, X = np.indices(data.shape)
	_, _, lamb = mos.meta.wcs.transform("detector", "world", X, Y)
	wave = lamb[yExtract - n:yExtract +n+1].mean(axis=0)
	flux = data[yExtract - n:yExtract +n+1].mean(axis=0)
	dflux = np.sqrt((mos.err[yExtract - n:yExtract +n+1]**2).mean(axis=0)) * factor

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
	ax[0].hlines((yExtract-n, yExtract+n), 0, data.shape[1], color='r', linewidth=0.5, linestyle='dashed')
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

	# Make x1d file
	col_waves = fits.Column(array=wave, format='D', name='WAVELENGTH')
	col_fluxes = fits.Column(array=flux, format='D', name='FLUX')
	col_flux_errs = fits.Column(array=dflux, format='D', name='FLUX_ERR')
	cols = fits.ColDefs([col_waves, col_fluxes, col_flux_errs])

	name = file.replace("_s2d.fits", f"{extractId}_x1d.fits")
	logConsole(f"Saving {name}")
	tbhdu = fits.BinTableHDU.from_columns(cols)
	tbhdu.writeto(name, overwrite=True)

	name = file.replace("_s2d.fits", f"{extractId}_extracted.png")
	logConsole(f"Saving {name}")
	plt.savefig(name)
	plt.close()

def changeUnit(bunit,sr):
	if bunit == "MJy":
		unit = "Jy"
		_ = 1e6
	elif bunit == "MJy/sr":
		unit = "Jy"
		_ = sr * 1e6
	else:
		unit = bunit
		_ = 1

	return _, unit


#IterateOverFolders(folders)
makeExtraction("./mastDownload/JWST/CEERS-NIRSPEC-P11-PRISM-MSATA/Final/jw01345-o100_s03480_nirspec_clear-prism_s2d.fits")