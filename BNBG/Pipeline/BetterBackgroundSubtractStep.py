import os.path

from scipy.ndimage import generic_filter
from scipy.stats import median_abs_deviation

from BNBG.utils import *
import stdatamodels.jwst.datamodels as dm
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.resample import ResampleSpecStep
from scipy import interpolate
from jwst.master_background import nirspec_utils
from jwst.stpipe import Step


class BetterBackgroundStep(Step):
	"""
	 Creates a _s2d-BNBG.fits file from a _s2d file

	 Params
	 ---------
	name : str
		Path to the file to open. Must be a _s2d from Stage 3
	"""

	spec = '''
		useCheckpointFiles = boolean(default=True) # If the step will save precalibrations for later use
		radius = float(default=5) # The radius of extraction around the source
		crop = integer(default=3) # how many lines will be ignored on top and below the resampled 2D image
		interpolationKnots = float(default=0.1) # The fraction of the total amount of points which should be knots
		kernelHeight = integer(default=1), height of kernel to use for the median filtering. By default, will be 1px in spatial direction
		kernelWidth = integer(default=15), width of kernel to use for the median filtering. By default, will be 15px in spectral direction
		Nsigma = float(default=10), number of sigmas above which pixels in error and data will be masked after median filtering
	     '''

	class_alias = 'bnbg'

	def __init__(
			self,
			name=None,
			parent=None,
			config_file=None,
			_validate_kwds=True,
			**kws,
	):
		super().__init__(name, parent, config_file, _validate_kwds)
		self.raw = None
		self.background = None
		self.resampled = None
		self.useCheckpointFiles = None
		self.Nsigma = None
		self.kernelWidth = None
		self.kernelHeight = None
		self.interpolationKnots = None
		self.crop = None
		self.radius = None
		self.output_dir = ""

	# noinspection PyTypeChecker
	def process(self, s2d):
		"""
		Calculates an interpolated background and subtracts it

		Parameters
		----------
		s2d : a SlitModel or a path

		Returns
		-------
		The background subtracted data model
		"""

		# s2d can be either a datamodel or the path to a slit fits
		# either case, we need both info
		self.raw = dm.open(s2d)
		name = os.path.basename(self._input_filename)

		# Getting background
		directory = self.output_dir
		pathBkg = os.path.join(directory, name.replace("s2d", "bkg-BNBG"))
		if not os.path.exists(pathBkg) or not self.useCheckpointFiles:
			self.background = workOnSlitlet(self.resampled,
											pathBkg,
											radius=self.radius,
											crop=self.crop,
											n=self.interpolationKnots,
											kernelSize=(self.kernelHeight,self.kernelWidth),
											Nsigma=self.Nsigma)
		else:
			logConsole(f"Found {os.path.basename(pathBkg)}")
			self.background = dm.open(pathBkg)

		# Subtract background from original file
		# TODO : Make custom subtraction routine
		pathBNBG = os.path.join(self.output_dir, name.replace("s2d", "s2d-BNBG"))
		if not os.path.exists(pathBNBG) or not self.useCheckpointFiles:
			result = subtractBackground(self.raw, self.background, pathBNBG)
		else:
			logConsole(f"Found {os.path.basename(pathBNBG)}")
			result = dm.open(pathBNBG)

		return result

def workOnSlitlet(resampled, pathClean, radius=4, crop=3, n=0.1, kernelSize=(1,15), Nsigma=10):
	# For a given _s2d
	logConsole("Starting modeling background")
	fitted = modelBackgroundFromSlit(resampled, radius=radius, crop=crop, n=n, kernelSize=kernelSize, Nsigma=Nsigma)

	# Overwrite data with background fit
	resampled.data = fitted
	logConsole("Background Calculated!")

	logConsole("Saving Clean Background File...")
	resampled.write(pathClean)

	return resampled

def modelBackgroundFromSlit(slit, radius=4, crop=3, n=0.1, kernelSize=(1,15), Nsigma=10):
	"""
	Creates a 2D image model based on the wavelengths positions of the background

	Parameters
	----------
	slit : a slit object, It is assumed that it contains WCS data and that a resampling step has been applied
	radius : float, radius of mask around source
	crop : int, number of lines to ignore above and below
	n : float, fraction of total datapoints to be knots
	kernelSize : tuple, size of kernel to use for the median filtering. By default, will be 15 pixels wide in spectral direction and 1 in spatial direction
	Nsigma : float, number of sigmas above which pixels in error and data will be masked after median filtering

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
									radius=radius,
									crop=crop,
									n=n,
									kernelSize=kernelSize,
									Nsigma=Nsigma)

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
							 source = None,
							 radius = 4,
							 crop = 3,
							 n = 0.1,
							 modelImage = None,
							 kernelSize=(1,15),
							 Nsigma=10):
	# TODO : replace redundant args by a kwarg thingy
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	preCalibrationWavelength : np.ndarray, 2D array representing the wavelength at each pixel
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image
	source : float, the source position along the vertical axis, in pixels
	radius : float, radius of mask around source
	crop : int, number of lines to ignore above and below
	n : float, fraction of total datapoints to be knots
	modelImage : np.ndarray, 2D array representing the envelope-less and noiseless image. Used for testing
	kernelSize : tuple, size of kernel to use for the median filtering. By default, will be 15 pixels wide in spectral direction and 1 in spatial direction
	Nsigma : float, number of sigmas above which pixels in error and data will be masked after median filtering

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background, or zeros if a fit cannot be made
	"""
	x,y,dy = getDataWithMask(data, error, wavelength, source, radius, crop, modelImage, kernelSize=kernelSize, Nsigma=Nsigma)

	if x is None and y is None and dy is None:
		logConsole("No data was kept in slit. Returning zeros", "WARNING")
		return np.zeros_like(wavelength)

	# Check if at least 4 points
	if len(x) < 4:
		logConsole("Not enough points to fit. Returning zeros", "WARNING")
		return np.zeros_like(wavelength)

	# Weights, as a fraction of total sum, else it breaks the fitting
	w = 1/dy
	w /= w.mean()

	interp = makeInterpolation(x,y,w,n)
	# The 2D background model obtained from the 1D spectrum
	return interp(wavelength)

def getDataWithMask(data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 source = None,
							 radius = 4,
							 crop = 3,
							 modelImage = None,
							 kernelSize=(1,15),
							 Nsigma=10):
	"""
	Extracts 3 1D arrays x, y and dy from 2D arrays

	Parameters
	----------
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image
	source : float, the source position along the vertical axis, in pixels
	radius : float, radius of mask around source
	crop : int, number of lines to ignore above and below
	modelImage : np.ndarray, 2D array representing the envelope-less and noiseless image. Used for testing
	kernelSize : tuple, size of kernel to use for the median filtering. By default, will be 15 pixels wide in spectral direction and 1 in spatial direction
	Nsigma : float, number of sigmas above which pixels in error and data will be masked after median filtering

	Returns
	-------
	x, y, dy

	"""
	isModelValid = verifySimilarImages(data, modelImage)
	yModel = None

	mask = cleanupImage(data, error, source=source, radius=radius, crop=crop, kernelSize=kernelSize, Nsigma=Nsigma)
	if np.all(mask):
		# No data
		return None, None, None

	x = extractWithMask(wavelength, mask)
	y = extractWithMask(data, mask)
	dy = np.sqrt(extractWithMask(error ** 2, mask))

	nanMask = (np.isnan(y)) | (np.isnan(dy)) | (np.isnan(x))

	if isModelValid:
		yModel = extractWithMask(modelImage, mask)
		nanMask = nanMask | (np.isnan(yModel))
		yModel = yModel[~nanMask]

	x = x[~nanMask]
	y = y[~nanMask]
	dy = dy[~nanMask]

	# Sort arrays in rising x order
	indices = np.argsort(x)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]

	return x,y,dy

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

	# Regarding the values outside the range of wavelength :
	# Extreme wavelengths could be filtered by the masking, thus reducing the range
	# The default behavior of scipy.interpolate.UnivariateSpline is extrapolating the spline
	# This is usually not recommended if the value to extrapolate is too far away from the range
	# As we are however talking about a few pixels, this should normally not be an issue
	interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=b * len(x))
	N_b = len(interp.get_knots())
	N_a = len(x)

	# We want N = n*m knots
	# Since the backend of scipy uses a Fortran library which only allows for a smoothing factor or a given set of knots
	# We use a binary search to pinpoint the S value which gives a spline of N knots
	targetN = round(n * len(x))

	# Check if targetN is between the max amount of points and the first estimate N
	# Assign a higher S value if not
	if not N_a > targetN > N_b:
		b = 100 * len(x)
		interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=b * len(x))
		N_b = len(interp.get_knots())

	N_c = N_a
	iteration = 0
	while N_c != targetN and iteration < 100:
		# Sometimes even though a and b are identical, the number of knots is different
		# This is because the gods of interpolation actually hate me personally
		if a==b:
			break
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

	# This then removes slit by slit the background
	# TODO : No error is propagated, if the fitting ends up being rewritten to account for errors,
	#  this should be replaced with a custom made function
	result = nirspec_utils.apply_master_background(raw, background)

	result.write(pathBNBG)
	logConsole(f"Saving File {os.path.basename(pathBNBG)}")

	return result