from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from ..utils import *
from scipy.optimize import curve_fit as cfit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
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
	saveBackgroundImage : bool
		If true, will save the background image as a fits

	"""
	if not "_srctype" in name:
		logConsole(f"{name.split('/')[-1]} not a _srctype file. Skipping...",source="WARNING")
		pass

	# 1st draft Algorithm :
	logConsole(f"Starting Custom Background Subtraction on {name.split('/')[-1]}",source="BetterBackground")
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

		shutter_id = WhichShutterOpen(slit.shutter_state)
		if shutter_id is None:
			logConsole("Not a 3 shutter slit!")
			precal[i].data = np.zeros_like(precal.slits[i].data)
			continue

		logConsole("Starting on modeling of background")
		fitted = modelBackgroundFromSlit(slit,shutter_id,precal.slits[i])

		if saveBackgroundImage:
			z = ZScaleInterval()
			z1, z2 = z.get_limits(slit.data)
			plt.figure(figsize=(16,4))
			plt.subplot(2,1,1)
			plt.imshow(precal.slits[i].data,origin="lower",interpolation="none",vmin=z1,vmax=z2)
			plt.subplot(2, 1, 2)
			plt.imshow(fitted, origin="lower", interpolation="none", vmin=z1, vmax=z2)
			plt.savefig(f"{precal.slits[i].slitlet_id}.png")
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

def modelBackgroundFromSlit(slit, shutter_id, precalSlit):

	Y, X = np.indices(slit.data.shape)
	_, _, dataLambda = slit.meta.wcs.transform("detector", "world", X, Y)

	Y, X = np.indices(precalSlit.data.shape)
	_, _, precalLambda = precalSlit.meta.wcs.transform("detector", "world", X, Y)

	# The slices are NOT the size of the individual shutter dispersion
	# They are chosen by analysing the vertical maxima and choosing a radius of lines above and beyond those maxima

	return modelBackgroundFromImage(precalSlit.data,
									precalLambda,
									slit.data,
									slit.err,
									dataLambda,
									shutter_id)

def modelBackgroundFromImage(preCalibrationData : np.ndarray,
							 preCalibrationWavelength : np.ndarray,
							 data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 shutter_id : int):
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	preCalibrationData : np.ndarray, 2D array of the untreated image
	preCalibrationWavelength : np.ndarray, 2D array representing the wavelength at each pixel
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image
	shutter_id : int, ID of the shutter which contains the signal (0,1,2)

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background, or zeros if a fit cannot be made
	"""

	slice_indices = SelectSlice(data)


	##### TEST #####
	plt.figure(0)
	plt.hlines(np.array(slice_indices).ravel(), 0, data.shape[1], color='r')
	plt.xlim(data.shape[0], data.shape[1])
	################

	if slice_indices is None:
		logConsole("Data is empty")
		return np.zeros_like(preCalibrationData)

	x = extract1DBackgroundFromImage(wavelength, slice_indices, shutter_id)
	y = extract1DBackgroundFromImage(data,slice_indices,shutter_id)
	dy = extract1DBackgroundFromImage(error**2,slice_indices,shutter_id)

	mask = ~np.logical_or(
		np.logical_or(
			np.isnan(x),
			np.isnan(y)
		),
		np.isnan(dy)
	)
	x = x[mask]
	y = y[mask]
	dy = dy[mask]

	_, indices = np.unique(x, return_index=True)
	indices = np.sort(indices)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]

	dy = np.sqrt(dy)

	# Sorts in rising wavelength order, ignores aberrant y values
	indices = np.argsort(x)
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
		logConsole("Not Enough Points to interpolate", source="WARNING")
		return np.zeros_like(preCalibrationData)

	interp = makeInterpolation(x,y,w)

	# The 2D background model obtained from the 1D spectrum
	return interp(preCalibrationWavelength)

def extract1DBackgroundFromImage(data : np.ndarray, slice_indices : iter, shutter_id : int) -> np.ndarray:
	"""
	Sums vertically the image on 2 horizontal slices and appends the 2
	Parameters
	----------
	data : 2D array, image
	slice_indices : 3x2 iterable, the 1st index is the n° of the slice, the 2nd is the [start,end] index
	shutter_id : int, 0 is for the top slice, 1 for the middle, 2 for the bottom

	Returns
	-------
	1D array, wavelength dependant

	"""
	return np.append(data[slice_indices[shutter_id - 1][0]:slice_indices[shutter_id - 1][1]].mean(axis=0),
				  data[slice_indices[shutter_id - 2][0]:slice_indices[shutter_id - 2][1]].mean(axis=0))


def makeInterpolation(x : np.ndarray, y : np.ndarray, w : np.ndarray):
	"""
	Creates a spline interpolation / approximation of order 5

	Parameters
	----------
	x : wavelength 1D array
	y : corresponding data 1D array
	w : weights of the data points, inverse of their error

	Returns
	-------
	interp : a function which approximates the data
	"""
	mean = []
	std = []
	n = 20
	for j,S in enumerate(10**np.linspace(-8,12,n)):
		interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=S*len(w))
		Y = interp(x)
		mean.append(np.mean(abs(Y-y)))
		std.append(np.std(Y-y))
		real = interp1d([0,2000,400,1300],[200,10,140,100],kind='cubic')

		if j%2 == 0:
			plt.figure()
			plt.plot(x,Y, color='b')
			plt.scatter(x,y, color='k', marker='+')
			plt.plot(x,real(x), color='r')

			plt.title(f"S = {S}")

	plt.figure()
	plt.plot(10**np.linspace(-8,12,n), mean, label="mean",marker='+')
	plt.hlines(0, 10 ** -8, 10 ** 12, color='r', linestyle='--')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel("mean")
	plt.xlabel("S")


	plt.figure()
	plt.plot(10 ** np.linspace(-8, 12, n), std, label="std",marker='+')
	plt.hlines(20,10**-8,10**12,color='r',linestyle='--')
	plt.yscale('log')
	plt.xscale('log')
	plt.ylabel(r"\sigma")
	plt.xlabel("S")

	plt.show()
	return interp


def SelectSlice(slitData : np.ndarray) :
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross-section, finding a pixel position for each peak,
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	slitData, 2D array, is supposed to be aligned with the pixel grid, so a resampledStep is needed before that

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit

	"""
	# The radius of each slice
	radius = 10
	data = sigma_clip(slitData, sigma=5, masked=True)

	# Get vertical cross-section by summing horizontally
	horiz_sum = data.mean(axis=1)

	# Determine 3 maxima for 3 slits
	peaks = []
	# Looks for peaks of different width
	j = 2
	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1

	# Gets equal slices if no peaks are found
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
		return np.array([xmin, xmax]).T

	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(np.array(range(len(horiz_sum))),horiz_sum,peaks))

	return np.clip(getSliceFromPeaks(peaks, radius), 0, slitData.shape[0])


def getPeaksPrecise(x : np.ndarray, y : np.ndarray, peaks) -> np.ndarray:
	"""
	Gets a subpixel position for each peak
	Parameters
	----------
	x : wavelength dependant, 1D array
	y : value, 1D array
	peaks : wavelength position of the peaks, list

	Returns
	-------
	array of peak positions, subpixel if a gaussian fit was found, integer if not
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5,0],full_output=True)
	except :
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return np.array(peaks)
	return np.array(coeff[:3])


def getSliceFromPeaks(peaks, n : int) -> np.ndarray:
	"""
	Gets the indices for 3 slices around the peaks
	Parameters
	----------
	peaks : list of peak positions
	n : radius

	Returns
	-------
	a 2x3 array of indices of 3 slices

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