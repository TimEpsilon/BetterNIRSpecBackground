import numpy as np
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from utils import logConsole
from astropy.visualization import ZScaleInterval


working_dir = "./mastDownload/JWST/"

for folders in glob(working_dir+'*'):
	path = folders + "/Final/"
	if os.path.exists(path):
		logConsole(f"Working on {folders}")
		for file in glob(path+"*_s2d.fits"):
			logConsole(f"Working on {file}")
			with dm.open(file) as mos:
				if len(mos["shutter_state"]) != 3:
					continue

				# The position of the source
				# The slice will be made around this
				s_dec, s_ra = mos["source_dec"], mos["source_ra"]

				# Map of WCS
				Y, X = np.indices(mos["data"].shape)
				wcs = mos["meta"].wcs.transform('detector', 'world', X, Y)
				ra, dec, lamb = wcs

				z = ZScaleInterval()
				z1, z2 = z.get_limits(mos["data"])

				fig, ax = plt.subplots(2, figsize=(10, 6), tight_layout=True)
				im = ax[0].imshow(mos["data"], interpolation='None', origin="lower",vmin=z1,vmax=z2)
				divider = make_axes_locatable(ax[0])
				cax = divider.append_axes('right', size='5%', pad=0.05)
				fig.colorbar(im, cax=cax, orientation='vertical',label=f"Flux ({mos['meta'].bunit_data})")

				n = 3
				x, y = mos["meta"].wcs.transform("world", "detector", s_ra, s_dec, lamb[0][0])
				ax[0].hlines((y + n, y - n), 0, Y.shape[1], color='r', linewidth=0.5, linestyle='dashed')

				wavelength = lamb[round(y) - n:round(y) + n, :]
				target = mos["data"][round(y) - n:round(y) + n, :]
				err = mos["err"][round(y) - n:round(y) + n, :]

				ra = ra.mean(axis=1)
				dec = dec.mean(axis=1)

				wavelength = wavelength.mean(axis=0)
				spectrum = target.mean(axis=0)
				err = np.mean(err ** 2, axis=0) + target.var(axis=0)
				err = np.sqrt(err)

				ax[1].errorbar(wavelength, spectrum, yerr=err, marker='.', color="k", linewidth=0.3)
				ax[1].grid(True)

				name = file.split("/")[-1].replace("_s2d.fits",".png")
				logConsole(f"Saving {name}")
				plt.savefig(file.replace("_s2d.fits", ".png"))