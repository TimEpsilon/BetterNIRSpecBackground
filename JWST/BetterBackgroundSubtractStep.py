import numpy as np
from scipy.signal import find_peaks_cwt
from utils import *
from scipy.optimize import curve_fit as cfit
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from scipy.ndimage import rotate
from astropy.stats import sigma_clip
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.resample import ResampleSpecStep
from scipy import interpolate
from jwst.master_background import nirspec_utils
from astropy.visualization import ZScaleInterval


def BetterBackgroundStep(name,saveBackgroundImage=False):
	"""
	 Creates a _bkg file from a _srctype file

	 Params
	 ---------
	 name : str
		Path to the file to open. Must be a _srctype
	"""
	if not "_srctype" in name:
		logConsole(f"{name.split('/')[-1]} not a _srctype file. Skipping...",source="WARNING")
		pass

	# 1st draft Algorithm :
	logConsole(f"Starting Custom Background Substraction on {name.split('/')[-1]}",source="BetterBackground")
	multi_hdu = dm.open(name)
	logConsole("Applying Pre-Calibration...")

	precal = FlatFieldStep.call(multi_hdu)
	precal = PathLossStep.call(precal,source_type="EXTENDED")
	precal = BarShadowStep.call(precal,source_type="EXTENDED")
	precal = PhotomStep.call(precal,source_type="EXTENDED")

	# This is only useful for the extraction of the 1D spectrum
	resampled = ResampleSpecStep.call(precal)

	# For a given _srctype, for every slit
	for i,slit in enumerate(resampled.slits):
		logConsole(f"Opened slitlet {slit.slitlet_id}")

		#TODO : Eventually, work on error propagation

		shutter_id = WhichShutterOpen(slit.shutter_state)
		if shutter_id is None:
			logConsole("Not a 3 shutter slit!")
			precal[i].data = np.zeros_like(precal[i].data)
			continue

		# The slices are NOT the size of the individual shutter dispersion
		# They are chosen by analysing the vertical maxima and choosing a radius of lines above and beyond those maxima
		slice_indices = SelectSlice(slit)
		if slice_indices is None:
			logConsole("Data is empty")
			precal.slits[i].data = np.zeros_like(precal.slits[i].data)
			continue
		slice_indices = np.clip(slice_indices,0,slit.data.shape[0])

		logConsole("Starting on modeling of background")
		Y, X = np.indices(slit.data.shape)
		_, _, lamb = slit.meta.wcs.transform("detector", "world", X, Y)

		x = np.append(lamb[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]].mean(axis=0),
					  lamb[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]].mean(axis=0))

		y = np.append(slit.data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]].mean(axis=0),
					  slit.data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]].mean(axis=0))

		dy = np.append(slit.data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]].std(axis=0),
					   slit.data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]].std(axis=0))
		dy = dy**2
		dy += np.append((slit.err[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]]**2).mean(axis=0),
						(slit.err[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]]**2).mean(axis=0))
		dy = np.sqrt(dy)

		indices = np.argsort(x)

		# Sorts in rising wavelength order, ignores aberrant y values
		x = x[indices]
		y = y[indices]
		dy = dy[indices]
		x = x[y > 0]
		dy = dy[y > 0]
		y = y[y > 0]

		# Weights, as a fraction of total sum, else it breaks the fitting
		w = 1 / dy
		w /= w.mean()

		if len(x) <= 5:
			logConsole("Not Enough Points to interpolate",source="WARNING")
			precal.slits[i].data = np.zeros_like(precal.slits[i].data)
			continue

		# The s value should usually not cause the fitting to fail
		# In the case it does, a larger, less harsh s value is used
		# This is done by verifying if the returned function is nan on one of the 10 points in the wavelength range
		interp = interpolate.UnivariateSpline(x, y, w=w, s=0.01, k=5, check_finite=True)
		_ = interp(np.linspace(x.min(), x.max(), 10))
		if not np.all(np.isfinite(_)):
			for s in [0.01,0.1,1,10,100]:
				logConsole(f"Ideal spline not found. Defaulting to a spline of s={s}",source="WARNING")
				interp = interpolate.UnivariateSpline(x, y, w=w, s=s, k=3, check_finite=True)
				_ = interp(np.linspace(x.min(), x.max(), 10))
				if np.all(np.isfinite(_)):
					break

		# The 2D background model obtained from the 1D spectrum
		Y, X = np.indices(precal[i].data.shape)
		_, _, lamb = precal[i].meta.wcs.transform("detector", "world", X, Y)
		fitted = interp(lamb)

		if saveBackgroundImage:
			z = ZScaleInterval()
			z1, z2 = z.get_limits(slit.data)
			plt.figure(figsize=(16,4))
			plt.subplot(2,1,1)
			plt.imshow(precal[i].data,origin="lower",interpolation="none",vmin=z1,vmax=z2)
			plt.subplot(2, 1, 2)
			plt.imshow(fitted, origin="lower", interpolation="none", vmin=z1, vmax=z2)
			plt.savefig(f"{precal[i].slitlet_id}.png")
			plt.close()

		precal.slits[i].data = fitted
		logConsole("Background Calculated!")

	# Reverse the calibration
	logConsole("Reversing the Pre-Calibration...")
	background = PhotomStep.call(precal, inverse=True, source_type="EXTENDED")
	background = BarShadowStep.call(background, inverse=True, source_type="EXTENDED")
	background = PathLossStep.call(background, inverse=True, source_type="EXTENDED")
	background = FlatFieldStep.call(background, inverse=True)

	# The result is somehow inverted?
	background.write(name.replace("srctype","background"))
	result = nirspec_utils.apply_master_background(multi_hdu, background)

	multi_hdu.close()
	del multi_hdu
	del background
	del resampled
	del precal

	result.write(name.replace("srctype","bkg"))
	logConsole(f"Saving File {name.replace('srctype','bkg').split('/')[-1]}")

	pass


def SelectSlice(slit):
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross section, finding a pixel position for each peak,
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	slit, is supposed to be aligned with the pixel grid, so a resampledStep is needed before that

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit
	rotated : tuple of arrays

	"""
	# The radius of each slice
	radius = 1

	data = sigma_clip(slit.data, sigma=5, masked=True)

	# Get vertical cross section by summing horizontally
	horiz_sum = data.mean(axis=1)

	# Determine 3 maxima for 3 slits
	peaks = []
	j = 2

	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1
	if not len(peaks) == 3 or np.any(peaks > len(horiz_sum)) or np.any(peaks < 0):
		logConsole("Can't find 3 spectra. Defaulting to equal slices", source="WARNING")
		# Somehow this edge case exists
		if np.all(horiz_sum == 0):
			return None
		start_index = np.where(horiz_sum > 0)[0][0]
		end_index = np.where(horiz_sum > 0)[0][-1]
		n = end_index - start_index
		xmin = np.array([round(n / 6)-radius, round(n/2)-radius, round(5*n/6)-radius]) + start_index
		xmax = np.array([round(n / 6)+radius, round(n/2)+radius, round(5*n/6)+radius]) + start_index + 1
		slice_indices = np.array([xmin, xmax]).T
		return slice_indices

	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(range(len(horiz_sum)),horiz_sum,peaks))


	slice_indices = getPeakSlice(peaks,radius)

	return slice_indices


def getPeaksPrecise(x,y,peaks):
	"""
	Returns the sub-pixel peaks
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5,0],full_output=True)
	except :
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return peaks
	return np.array(coeff[:3])


def getPeakSlice(peaks,n):
	"""
	Returns slices for a set of peaks
	"""

	# Slice radius
	xmin = np.array([
		round(peaks[0]-n),
		round(peaks[1]-n),
		round(peaks[2]-n)])

	xmax = np.array([
		round(peaks[0]+n+1),
		round(peaks[1]+n+1),
		round(peaks[2]+n+1)])

	return np.array([xmin,xmax]).T

#BetterBackgroundStep("./mastDownload/JWST/CEERS-NIRSPEC-P5-PRISM-MSATA/jw01345063001_03101_00001_nrs1_srctype.fits",True)