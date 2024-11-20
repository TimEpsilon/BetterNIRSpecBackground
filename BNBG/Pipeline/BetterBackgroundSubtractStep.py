import os.path

import matplotlib
import numpy as np

matplotlib.use('TkAgg')

from BNBG.utils import *
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
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
	 Creates a _BNBG file from a _srctype file

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

	logConsole(f"Starting Custom Background Subtraction on {name.split('/')[-1]}",source="BetterBackground")
	multi_hdu = dm.open(name)
	logConsole("Applying Pre-Calibration...")

	if not os.path.exists(name.replace("srctype", "photomstep")):
		precal = FlatFieldStep.call(multi_hdu)
		precal = PathLossStep.call(precal,source_type="EXTENDED")
		precal = BarShadowStep.call(precal,source_type="EXTENDED")
		precal = PhotomStep.call(precal,source_type="EXTENDED")

		logConsole("Saving Photometry File...")
		precal.write(name.replace("srctype", "photomstep"))
	else :
		logConsole(f"Found {name.replace('srctype', 'photomstep')}")
		precal = dm.open(name.replace("srctype", "photomstep"))

	if not os.path.exists(name.replace("srctype", "resamplespecstep")):
		# This is only useful for the extraction of the 1D spectrum
		resampled = ResampleSpecStep.call(precal)

		logConsole("Saving Resampling File...")
		resampled.write(name.replace("srctype", "resamplespecstep"))
	else:
		logConsole(f"Found {name.replace('srctype', 'resamplespecstep')}")
		resampled = dm.open(name.replace("srctype", "resamplespecstep"))

	# For a given _srctype, for every slit
	for i,slit in enumerate(resampled.slits):
		logConsole(f"Opened slitlet {slit.slitlet_id}")
		logConsole("Starting on modeling of background")
		fitted = modelBackgroundFromSlit(slit,precal.slits[i])

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

		# Overwrite data with background fit
		precal.slits[i].data = fitted
		logConsole("Background Calculated!")

	precal.write(name.replace("srctype", "clean_background"))
	# Reverse the calibration
	logConsole("Reversing the Pre-Calibration...")
	background = PhotomStep.call(precal, inverse=True, source_type="EXTENDED")
	background = BarShadowStep.call(background, inverse=True, source_type="EXTENDED")
	background = PathLossStep.call(background, inverse=True, source_type="EXTENDED")
	background = FlatFieldStep.call(background, inverse=True)

	# The result is somehow inverted?
	background.write(name.replace("srctype","background"))
	# This then removes slit by slit the background
	# TODO : check how the error is propagated
	result = nirspec_utils.apply_master_background(multi_hdu, background)

	multi_hdu.close()
	del multi_hdu
	del background
	del resampled
	del precal

	result.write(name.replace("srctype","BNBG"))
	logConsole(f"Saving File {name.replace('srctype','BNBG').split('/')[-1]}")

	pass

def modelBackgroundFromSlit(slit, precalSlit):
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	slit : a slit object, It is assumed that it contains WCS data and that a resampling step has been applied
	precalSlit : a slit object, the same slit just before the resampling step

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background
	"""
	Y, X = np.indices(slit.data.shape)
	_, _, dataLambda = slit.meta.wcs.transform("detector", "world", X, Y)

	Y, X = np.indices(precalSlit.data.shape)
	_, _, precalLambda = precalSlit.meta.wcs.transform("detector", "world", X, Y)

	source = getSourcePosition(slit)

	return modelBackgroundFromImage(precalLambda,
									slit.data.copy(),
									slit.err.copy(),
									dataLambda,
									source=source)

def modelBackgroundFromImage(preCalibrationWavelength : np.ndarray,
							 data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 source = None,
							 modelImage = None):
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	preCalibrationWavelength : np.ndarray, 2D array representing the wavelength at each pixel
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image
	source : float, the source position along the vertical axis, in pixels
	modelImage : np.ndarray, 2D array representing the envelope-less and noiseless image. Used for testing

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background, or zeros if a fit cannot be made
	"""
	isModelValid = verifySimilarImages(data, modelImage)
	yModel = None

	mask = cleanupImage(data, source=source)
	if np.all(mask):
		logConsole("No data was kept in slit. Returning zeros","WARNING")
		return np.zeros_like(preCalibrationWavelength)

	x = extractWithMask(wavelength, mask)
	y = extractWithMask(data, mask)
	w = 1/np.sqrt(extractWithMask(error**2, mask))

	nanMask = (np.isnan(y)) | (np.isnan(w)) | (np.isnan(x))

	if isModelValid:
		yModel = extractWithMask(modelImage, mask)
		nanMask = nanMask | (np.isnan(yModel))
		yModel = yModel[~nanMask]

	x = x[~nanMask]
	y = y[~nanMask]
	w = w[~nanMask]

	# Sort arrays in rising x order
	indices = np.argsort(x)
	x = x[indices]
	y = y[indices]
	w = w[indices]

	# Weights, as a fraction of total sum, else it breaks the fitting
	w /= w.mean()

	# Check if at least 4 points
	if len(x) < 4:
		logConsole("Not enough points to fit. Returning zeros", "WARNING")
		return np.zeros_like(preCalibrationWavelength)

	interp = makeInterpolation(x,y,w)
	# The 2D background model obtained from the 1D spectrum
	return interp(preCalibrationWavelength)

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

def makeInterpolation(x: np.ndarray, y: np.ndarray, w: np.ndarray, n = 0.1):
	"""
    Creates a spline interpolation / approximation of order 3.

    Parameters
    ----------
    x : wavelength 1D array
    y : corresponding data 1D array
    w : weights of the data points, inverse of their error
    n : float, with N the number of knots and m the number of points, n = N/m

    Returns
    -------
    interp : a function which approximates the data
    """
	# S is defined as S = s / len(x), needs to be tweaked for a smoother fit
	a = 0
	b = 1
	interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=b * len(x))
	N_b = len(interp.get_knots())
	N_a = len(x)

	# We want N = n*m knots
	# Since the backend of scipy uses a Fortran library which only allows for a smoothing factor or a given set of knots
	# We use a binary search to pinpoint the S value which gives a spline of N knots
	targetN = round(n*len(x))

	# Check if targetN is between the max amount of points and the first estimate N
	# Assign a higher S value if not
	if not N_a > targetN > N_b:
		b = 100 * len(x)
		interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=b * len(x))
		N_b = len(interp.get_knots())

	N_c = N_a
	iteration = 0
	while N_c != targetN and iteration < 100:
		iteration += 1
		c = (a+b)/2
		interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=c * len(x))
		N_c = len(interp.get_knots())
		if N_a > targetN > N_c:
			b = c
			interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=b * len(x))
			N_b = len(interp.get_knots())
		else:
			a = c
			interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=a * len(x))
			N_a = len(interp.get_knots())

	logConsole(f"Finished Interpolation with {iteration} iterations, found S = {a*len(x)}")
	interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=a * len(x), ext=1)
	return interp

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

def cleanupImage(data : np.ndarray, crop=3, source=None, radius=5):
	"""
	Creates a mask that selects bad pixels for background subtraction
	Parameters
	----------
	data : 2D array
	crop : float, the amount of pixels to crop above and below the image
	source : float, the vertical position, in number of lines, from which to mask the source. If None, no source will be masked
	radius : float, radius of extraction along the source

	Returns
	-------
	mask : 2D array, array of boolean where True means the pixel was kept for background subtraction and False means it was rejected

	"""
	Y, X = np.indices(data.shape)
	# Gets rid of negative values, crops the top and bottom of the image, ignores pixels marked as nan
	mask = (data <= 0) | (Y < crop) | (Y > Y.max() - crop) | np.isnan(data)

	# If source in the image, remove lines in a radius around
	# This also works if the source is not in frame and the returned position is 1e48
	if source is not None:
		mask = mask | (np.round(abs(Y - source)) < radius)

	# TODO ? : Case of secondary source?

	return mask