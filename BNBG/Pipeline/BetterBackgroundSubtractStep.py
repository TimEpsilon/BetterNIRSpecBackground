import inspect
import os.path

from astropy.visualization import ZScaleInterval

from BNBG.utils import logConsole, getSourcePosition, PathManager, getCRDSPath

os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.stats import median_abs_deviation
import numpy as np
from stdatamodels.jwst.datamodels import MultiSlitModel
from jwst.barshadow import BarShadowStep

from BNBG.Pipeline.BSplineLSQ import BSplineLSQ

def BetterBackgroundStep(ratePath,
						 s2d,
						 cal,
						 radius=5,
						 crop=3,
						 interpolationKnots=0.1,
						 curvatureConstraint=0.5,
						 endpointConstraint=0.1,
						 kernelSize=(1,15),
						 Nsigma=10):
	"""
	 Creates a _s2d-BNBG.fits file from a _s2d file

	 Parameters
	 ----------
	 ratePath : PathManager
	 	Path based on initial rate file created at the start of Stage 2

	 s2d : MultiSlitModel
	 	DataModel post Stage 2. Must be a _s2d

	 cal : MultiSlitModel
	 	Datamodel post Stage2. Must be a _cal

	 radius : float
	 	The radius of extraction around the source

	 crop : int
	 	How many lines will be ignored on top and below the resampled 2D image

	 interpolationKnots : float
	 	The fraction of the total amount of points which should be knots

	 curvatureConstraint : float
	 	An hyperparameter used for regularizing, 
	 	(how much the curvature will be minimized).
	 	Useful if gaps are present in the data.
	 	If equal to 0, this will entirely ignore curvature

	 endpointConstraint : float
	 	An hyperparameter used for the endpoints
	 	(how much the slope on each side of the spline will be minimized).
	 	If equal to 0, this will entirely ignore the endpoint slopes.
	 	
	 kernelSize : tuple
	 	size of kernel to use for the median filtering
	 	
	 Nsigma : float
	 	number of sigmas above which pixels in error and data will be masked after median filtering

	 Returns
	 -------
	 results : MultiSlitModel
	 	The background MultiSlitModel, where each slit contains the corresponding background and error

	"""
	# The s2d model only serves for the extraction
	# No modification to it should be saved
	# Contrary to the master_background_mos step in the pipeline, which precalibrates each slit with the steps
	# FlatField, Pathloss, Barshadow, Photom in EXTENDED source type, we will only apply Barshadow,
	# as it is an almost purely spatial effect, and is usually skipped for point sources
	# This allows to get rid of the separation between the slits and helps for the extraction
	# TODO : This appears to break in a multiprocessing environment
	def barshadow(s2dInput):
		step = BarShadowStep()
		step.source_type = "EXTENDED"
		result = step.run(s2dInput)
		logConsole("Saving clean s2d")
		result.save(ratePath.withSuffix("s2d-BNBG"))
		return result

	s2d = ratePath.openSuffix("s2d-BNBG", lambda : barshadow(s2d))

	# Calculate background
	logConsole("Calculating Background")
	background = ratePath.openSuffix("bkg-BNBG", lambda : process(s2d,
																  cal,
																  ratePath.withSuffix("bkg-BNBG"),
																  radius=radius,
																  crop=crop,
																  n=interpolationKnots,
																  kernelSize=kernelSize,
																  Nsigma=Nsigma,
																  curvatureConstraint=curvatureConstraint,
																  endpointConstraint=endpointConstraint))

	def invBarshadow(bkgInput):
		# The Barshadow step now needs to be reversed
		step = BarShadowStep()
		step.source_type = "EXTENDED"
		step.inverse = True
		result = step.run(bkgInput)
		logConsole("Saving clean background")
		result.save(ratePath.withSuffix("cleanBkg-BNBG"))
		return result

	background = ratePath.openSuffix("cleanBkg-BNBG", lambda : invBarshadow(background))

	# Subtract background from original
	logConsole("Subtracting Background")
	result = ratePath.openSuffix("cal-BNBG", lambda : subtractBackground(cal,
																		 background,
																		 ratePath.withSuffix("cal-BNBG")))

	return result

def process(s2d, cal, pathClean, **kwargs):
	"""
	Calculates background per slit for a MultiSlitModel

	Parameters
	----------
	s2d : MultiSlitModel
		A _s2d datamodel (resampling has happened).

	cal : MultiSlitModel
		A _cal datamodel (output of Stage 2)

	pathClean : str
		Path to save the resulting file

	**kwargs :
		Arguments to pass to getDataWithMask and BSplineLSQ

	Returns
	-------
	results : MultiSlitModel
		a copy of ``cal``, for each slit,
		slit.data and slit.err correspond to the modelled background, and it's error.
		This is the 'pure' spectral background, with no spatial effects.

	"""
	for i, slit in enumerate(s2d.slits):
		logConsole("Calculating Background")
		s2dSlit = slit
		calSlit = cal.slits[i]

		Y, X = np.indices(calSlit.data.shape)
		_, _, targetLambda = calSlit.meta.wcs.transform("detector", "world", X, Y)
		Y, X = np.indices(s2dSlit.data.shape)
		_, _, dataLambda = s2dSlit.meta.wcs.transform("detector", "world", X, Y)

		source = getSourcePosition(slit)

		model, error = modelBackgroundFromImage(s2dSlit.data.copy(),
												s2dSlit.err.copy(),
												dataLambda,
												targetLambda,
												source=source,
												**kwargs)

		if model is None:
			calSlit.data = np.zeros_like(calSlit.data)
			# calSlit.err remains unchanged
		else:
			calSlit.data, calSlit.err = model, error

		logConsole("Background Calculated!")

	logConsole("Saving Clean Background File...")
	cal.save(pathClean)

	return cal

def modelBackgroundFromImage(data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 targetWavelength : np.ndarray,
							 **kwargs):
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	data : np.ndarray
		2D array of the s2d image

	error : np.ndarray
		2D array of the error of the s2d image

	wavelength : np.ndarray
		2D array of the wavelengths of the s2d image

	targetWavelength : np.ndarray
		2D array of the wavelengths of the cal image

	**kwargs :
		Arguments to pass to getDataWithMask and BSplineLSQ

	Returns
	-------
	results : (np.ndarray, np.ndarray)
		2D array of a smooth model of background and 2D array of its error, or zeros if a fit cannot be made
	"""

	# Getting 1D arrays
	kwargs_getDataWithMask = {k: v for k, v in kwargs.items() if k in inspect.signature(getDataWithMask).parameters}
	x,y,dy = getDataWithMask(data.copy(), error.copy(), wavelength.copy(), **kwargs_getDataWithMask)

	# Check if enough data points (spline of order 4 needs at least 10 points)
	if (x is None and y is None and dy is None) or len(x) < 10:
		logConsole("Not enough points to fit. Returning zeros", "WARNING")
		return None, None

	# Weights
	w = 1/dy**2

	# Creating bspline object
	kwargs_makeInterpolation = {k: v for k, v in kwargs.items() if k in inspect.signature(BSplineLSQ).parameters}
	bspline = BSplineLSQ(x,y,w,**kwargs_makeInterpolation)

	if True:
		plt.figure(figsize=(14,2))
		z1, z2 = ZScaleInterval().get_limits(data)
		plt.imshow(data, cmap='plasma', vmin=z1, vmax=z2, origin='lower')
		fig, ax = plt.subplots(figsize=(14,6))
		s1,s2 = bspline.plot(ax)
		plt.show()


	# The 2D background model obtained from the 1D spectrum
	return bspline(targetWavelength), bspline.getError(targetWavelength)

def getDataWithMask(data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 **kwargs):
	"""
	Extracts 3 1D arrays x, y and dy from 2D arrays

	Parameters
	----------
	data : np.ndarray
		2D array of the s2d image

	error : np.ndarray
		2D array of the error of the s2d image

	wavelength : np.ndarray
		2D array representing the wavelength of the s2d image

	**kwargs :
		Arguments to pass to cleanupImage

	Returns
	-------
	x, y, dy : (np.ndarray, np.ndarray, np.ndarray)
		1D arrays, respectively wavelength, flux and error
	"""

	data = data
	wavelength = wavelength
	error = error

	mask = cleanupImage(data.copy(), error.copy(), **kwargs)
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
		* Mask non-physical values (<0, NaN), crop the top and bottom pixels (crop parameter)
		* If a source is specified, masks the lines within radius around the source
		* Using this temporary mask, apply a median filtering with a given kernelSize on the data and error arrays
		* Subtracts this median array to the original data and error and calculates the median absolute deviation from each
		* Adds to the mask every pixel outside the range [-Nsigma * MAD, Nsigma * MAD] for both arrays

	Parameters
	----------
	data : np.ndarray
		2D array of the s2d image

	error : np.ndarray
		2D array of the error of the s2d image

	crop : int
		The amount of pixels to crop above and below the image

	source : float
		The vertical position, in number of lines, from which to mask the source. If None, no source will be masked

	radius : float
		Radius of extraction along the source

	kernelSize : tuple
		Size of kernel to use for the median filtering.
		By default, will be 15 pixels wide in spectral direction and 1 in spatial direction

	Nsigma : float
		Number of sigmas above which pixels in error and data will be masked after median filtering

	Returns
	-------
	mask : np.ndarray
		Array of boolean where False means the pixel was kept for background subtraction and True means it was rejected
	"""
	data = data
	error = error

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

	Parameters
	----------
	data : np.ndarray
		A 2D image

	mask : np.ndarray
		A 2D mask of the same size as data

	Returns
	-------
	x : np.ndarray
		The mean along the vertical axis, ignoring NaNs
	"""
	data[mask] = np.nan
	x = np.nanmean(data,axis=0)
	return x

def subtractBackground(raw, background, pathBNBG):
	"""
	Subtracts the background data from raw data.
	Also adds the errors appropriately.

	Parameters
	----------
	raw : MultiSlitModel
		Base model from which we want to subtract the background

	background : MultiSlitModel
		A similar model, which will be used as the background to subtract

	pathBNBG : str
		Where to save the result

	Returns
	-------
	result : MultiSlitModel
		The subtracted background

	"""
	result = raw.copy()
	for i, slit in enumerate(result.slits):
		resultSlit = slit
		bkgSlit = background.slits[i]

		resultSlit.data -= bkgSlit.data
		resultSlit.err = np.sqrt((resultSlit.err**2 + bkgSlit.err**2)/2)

	result.save(pathBNBG)
	logConsole(f"Saving File {os.path.basename(pathBNBG)}")

	return result