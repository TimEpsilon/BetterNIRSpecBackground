import inspect
import os.path

import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.stats import median_abs_deviation

from BNBG.Pipeline.BSplineLSQ import BSplineLSQ
from BNBG.utils import *
import stdatamodels.jwst.datamodels as dm

def BetterBackgroundStep(s2d, directory,
						 useCheckpoint=True,
						 radius=5,
						 crop=3,
						 interpolationKnots=0.1,
						 curvatureConstraint=0.5,
						 endpointConstraint=0.1,
						 kernelSize=(1,15),
						 Nsigma=10):
	"""
	 Creates a _s2d-BNBG.fits file from a _s2d file

	 Params
	 ---------
	s2d : str
		Path to the file to open. Must be a _s2d from Stage 3
	directory : str
		Output directory
	useCheckpointFiles : bool
		If the step will save precalibrations for later use
	radius : float
		The radius of extraction around the source
	crop : int
		how many lines will be ignored on top and below the resampled 2D image
	interpolationKnots : float
		The fraction of the total amount of points which should be knots
	curvatureConstraint : float
			An hyperparameter used for regularizing, i.e., how much the curvature will be minimized. Useful if gaps are present in the data
			If equal to 0, this will entirely ignore curvature
	endpointConstraint : float
		An hyperparameter used for the endpoints, i.e., how much the slope on each side of the spline will be minimized.
		If equal to 0, this will entirely ignore the endpoint slopes
	kernelSize : tuple
		size of kernel to use for the median filtering
	Nsigma : float
		number of sigmas above which pixels in error and data will be masked after median filtering
	"""
	resampled = dm.open(s2d)

	# Getting background
	name = os.path.basename(s2d)
	pathBkg = directory + name.replace("s2d", "bkg-BNBG")
	if not os.path.exists(pathBkg) or not useCheckpoint:
		background = workOnSlitlet(resampled.copy(),
										pathBkg,
										radius=radius,
										crop=crop,
										n=interpolationKnots,
										kernelSize=kernelSize,
										Nsigma=Nsigma,
								   		curvatureConstraint=curvatureConstraint,
								   		endpointConstraint=endpointConstraint)
	else:
		logConsole(f"Found {os.path.basename(pathBkg)}")
		background = dm.open(pathBkg)

	# Subtract background from original file
	pathBNBG = directory + name.replace("s2d", "s2d-BNBG")
	if not os.path.exists(pathBNBG) or not useCheckpoint:
		result = subtractBackground(resampled, background, pathBNBG)
	else:
		logConsole(f"Found {os.path.basename(pathBNBG)}")
		result = dm.open(pathBNBG)

	return result

def workOnSlitlet(resampled, pathClean, **kwargs):
	# For a given _s2d
	logConsole("Starting modeling background")
	fitted, error = modelBackgroundFromSlit(resampled, **kwargs)

	# Overwrite data with background fit
	resampled.data = fitted
	resampled.err = error
	logConsole("Background Calculated!")

	logConsole("Saving Clean Background File...")
	resampled.save(pathClean)

	return resampled

def modelBackgroundFromSlit(slit, **kwargs):
	"""
	Creates a 2D image model based on the wavelengths positions of the background

	Parameters
	----------
	slit : a slit object, It is assumed that it contains WCS data and that a resampling step has been applied

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background
	"""
	Y, X = np.indices(slit.data.shape)
	_, _, dataLambda = slit.meta.wcs.transform("detector", "world", X, Y)

	source = getSourcePosition(slit)

	return modelBackgroundFromImage(slit.data.copy(),
									slit.err.copy(),
									dataLambda,
									source=source,
									**kwargs)

def getSourcePosition(slit):
	"""
	Returns the vertical position, in detector space (pixels), of the source
	Parameters
	----------
	slit : a slit object, it is assumed that assign_wcs has been applied beforehand

	Returns
	-------
	vertical source position in pixels, can be ~1e48 if the source is outside in the case of a 2 slit slitlet
	"""
	return slit.meta.wcs.transform('world', 'detector', slit.source_ra, slit.source_dec, 3)[1]

def modelBackgroundFromImage(data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 **kwargs):
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image

	Returns
	-------
	interp : np.ndarray,
		2D array of a smooth model of background, or zeros if a fit cannot be made
	error : np.ndarray,
		Corresponding error of the model
	"""
	kwargs_getDataWithMask = {k: v for k, v in kwargs.items() if k in inspect.signature(getDataWithMask).parameters}
	x,y,dy = getDataWithMask(data, error, wavelength, **kwargs_getDataWithMask)

	if x is None and y is None and dy is None:
		logConsole("No data was kept in slit. Returning zeros", "WARNING")
		return np.zeros_like(wavelength), error

	# Check if at least 4 points
	if len(x) < 4:
		logConsole("Not enough points to fit. Returning zeros", "WARNING")
		return np.zeros_like(wavelength), error

	# Weights, as a fraction of total sum, else it breaks the fitting
	w = 1/dy
	#w /= w.mean()

	kwargs_makeInterpolation = {k: v for k, v in kwargs.items() if k in inspect.signature(BSplineLSQ).parameters}
	bspline = BSplineLSQ(x,y,w,**kwargs_makeInterpolation)
	"""
	plt.figure()
	bspline.plot(plt.gca())

	plt.figure()
	plt.imshow(wavelength, origin='lower')
	plt.show()
	"""
	# The 2D background model obtained from the 1D spectrum
	return bspline(wavelength), bspline.getError(wavelength)

def getDataWithMask(data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 **kwargs):
	"""
	Extracts 3 1D arrays x, y and dy from 2D arrays

	Parameters
	----------
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image

	Returns
	-------
	x, y, dy

	"""
	data = data.copy()
	wavelength = wavelength.copy()
	error = error.copy()

	mask = cleanupImage(data, error, **kwargs)
	if np.all(mask):
		# No data
		return None, None, None

	x = extractWithMask(wavelength, mask)
	y = extractWithMask(data, mask)
	dy = np.sqrt(extractWithMask(error ** 2, mask))

	nanMask = (np.isnan(y)) | (np.isnan(dy)) | (np.isnan(x))

	x = x[~nanMask]
	y = y[~nanMask]
	dy = dy[~nanMask]

	# Sort arrays in rising x order
	indices = np.argsort(x)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]

	return x,y,dy

def cleanupImage(data : np.ndarray, error : np.ndarray, crop=3, source=None, radius=5, kernelSize=(1,15), Nsigma=10):
	"""
	Creates a mask that selects bad pixels for background subtraction

	The algorithm is as such:
	1) Mask non-physical values (<0, NaN), crop the top and bottom pixels (crop parameter)
	2) If a source is specified, masks the lines within radius around the source
	3) Using this temporary mask, apply a median filtering with a given kernelSize on the data and error arrays
	4) Subtracts this median array to the original data and error and calculates the median absolute deviation from each
	5) Adds to the mask every pixel outside the range [-Nsigma * MAD, Nsigma * MAD] for both arrays

	Parameters
	----------
	data : 2D array
	error : 2D array, the associated error array
	crop : float, the amount of pixels to crop above and below the image
	source : float, the vertical position, in number of lines, from which to mask the source. If None, no source will be masked
	radius : float, radius of extraction along the source
	kernelSize : tuple, size of kernel to use for the median filtering. By default, will be 15 pixels wide in spectral direction and 1 in spatial direction
	Nsigma : float, number of sigmas above which pixels in error and data will be masked after median filtering

	Returns
	-------
	mask : 2D array, array of boolean where False means the pixel was kept for background subtraction and True means it was rejected
	"""
	data = data.copy()
	error = error.copy()

	Y, X = np.indices(data.shape)
	# Gets rid of negative values, crops the top and bottom of the image, ignores pixels marked as nan
	mask = (data <= 0) | (Y < crop) | (Y > Y.max() - crop) | np.isnan(data)

	# If source in the image, remove lines in a radius around
	# This also works if the source is not in frame and the returned position is 1e48
	if source is not None:
		# TODO ? : Case of secondary source?
		mask = mask | (np.round(abs(Y - source)) < radius)

	data[mask] = np.nan
	error[mask] = np.nan
	medianData = generic_filter(data, lambda x : np.nanmedian(x), size=kernelSize, mode="nearest")
	medianError = generic_filter(error, lambda x : np.nanmedian(x), size=kernelSize, mode="nearest")

	medianSubtractData = data - medianData
	_ = medianSubtractData.ravel()
	_ = _[np.isfinite(_)]
	MADData = median_abs_deviation(_)

	medianSubtractError = error - medianError
	_ = medianSubtractError.ravel()
	_ = _[np.isfinite(_)]
	MADError = median_abs_deviation(_)

	mask = mask | (np.abs(medianSubtractData) > Nsigma*MADData) | (np.abs(medianSubtractError) > Nsigma*MADError)

	return mask

def extractWithMask(data, mask):
	"""
	Gets a 1D array extraction of every value in data not masked by mask.
	----------
	data : ndarray, a 2D image
	mask : ndarray, a 2D mask of the same size as data

	Returns
	-------
	interp : a function of wavelength
	"""
	data[mask] = np.nan
	x = np.nanmean(data,axis=0)

	return x

def subtractBackground(raw, background, pathBNBG):
	"""
	Subtracts the background data from raw data, assuming a single slitlet.
	Also adds the errors appropriately.
	Parameters
	----------
	raw
	background
	pathBNBG

	Returns
	-------

	"""
	result = raw.copy()
	result.data -= background.data
	result.err = np.sqrt((result.err**2 + background.err**2)/2)

	result.save(pathBNBG)
	logConsole(f"Saving File {os.path.basename(pathBNBG)}")

	return result