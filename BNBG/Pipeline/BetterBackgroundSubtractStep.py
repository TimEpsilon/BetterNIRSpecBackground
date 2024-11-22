import os.path

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


class BetterBackgroundStep(Step):
	"""
	 Creates a _BNBG file from a _srctype file

	 Params
	 ---------
	 name : str
		Path to the file to open. Must be a _srctype
	saveBackgroundImage : bool
		If true, will save the background image as a fits

	"""

	spec = '''
		useCheckpointFiles = boolean(default=True) # If the step will save precalibrations for later use
		radius = float(default=5) # The radius of extraction around the source
		crop = integer(default=3) # how many lines will be ignored on top and below the resampled 2D image
		interpolationKnots = float(default=0.1) # The fraction of the total amount of points which should be knots
	     '''

	class_alias = 'bnbg'

	# noinspection PyTypeChecker
	def process(self, srctype):
		"""
		Calculates an interpolated background and subtracts it

		Parameters
		----------
		srctype : a MultiSlitModel or a path

		Attributes
		----------
		useCheckpointFiles : boolean
			Will use the precalibration files and the resampled file if found in output_dir.
			If false, will calculate them and save them in output_dir.

		radius : float
			The vertical radius in pixels around the source as given by the fits keywords which will be ignored by the extraction

		crop : integer
			The number of lines on top and below the resampled 2D image which will be ignored by the extraction

		interpolationKnots : float
			Number of knots used for the spline approximation, expressed as a fraction of the total amount of points

		Returns
		-------
		The background subtracted data model
		"""

		# srctype can be either a datamodel or the path to a multislit fits
		# either case, we need both info
		self.raw = dm.open(srctype)
		name = os.path.basename(self._input_filename)

		# TODO : Photomstep for precalibration is same name as photomstep post-background subtraction

		pathPhotom = os.path.join(self.output_dir, name.replace("srctype", "photomstep"))
		if not os.path.exists(pathPhotom) or not self.saveCheckpointFiles:
			self.precalibrated = BetterBackgroundStep.precalibration(self.raw, pathPhotom)
		else:
			logConsole(f"Found {pathPhotom}")
			self.precalibrated = dm.open(pathPhotom)

		# This is only useful for the extraction of the 1D spectrum
		pathResample = os.path.join(self.output_dir, name.replace("srctype", "resamplespecstep"))
		if not os.path.exists(pathResample) or not self.saveCheckpointFiles:
			self.resampled = BetterBackgroundStep.resampling(self.precalibrated, pathResample)
		else:
			logConsole(f"Found {pathResample}")
			self.resampled = dm.open(name.replace("srctype", "resamplespecstep"))

		pathClean = os.path.join(self.output_dir, name.replace("srctype", "clean_background"))
		self.cleanBackground = BetterBackgroundStep.workOnSlitlet(self.resampled,
																  self.precalibrated,
																  pathClean,
																  radius=self.radius,
																  crop=self.crop,
																  n=self.interpolationKnots)

		pathBackground = os.path.join(self.output_dir, name.replace("srctype", "background"))
		self.background = BetterBackgroundStep.reversePrecalibration(self.cleanBackground, pathBackground)

		pathBNBG = os.path.join(self.output_dir, name.replace("srctype", "BNBG"))
		self.result = BetterBackgroundStep.subtractBackground(self.raw, self.background, pathBNBG)

		return self.result




	@staticmethod
	def precalibration(srctype, pathPhotom):
		logConsole("Applying Pre-Calibration...")

		precalibration = FlatFieldStep.call(srctype)
		precalibration = PathLossStep.call(precalibration,source_type="EXTENDED")
		precalibration = BarShadowStep.call(precalibration,source_type="EXTENDED")
		precalibration = PhotomStep.call(precalibration,source_type="EXTENDED")

		logConsole("Saving Photometry File...")
		precalibration.write(pathPhotom)

		return precalibration

	@staticmethod
	def resampling(precalibration, pathResample):

		resampled = ResampleSpecStep.call(precalibration)

		logConsole("Saving Resampling File...")
		resampled.write(pathResample)

		return resampled

	@staticmethod
	def workOnSlitlet(resampled, precalibration, pathClean, radius=4, crop=3, n=0.1):
		# For a given _srctype, for every slit
		for i,slit in enumerate(resampled.slits):
			logConsole(f"Opened slitlet {slit.slitlet_id}")
			logConsole("Starting on modeling of background")
			fitted = BetterBackgroundStep.modelBackgroundFromSlit(slit, precalibration.slits[i], radius=radius, crop=crop, n=n)

			# Overwrite data with background fit
			precalibration.slits[i].data = fitted
			logConsole("Background Calculated!")

		logConsole("Saving Clean Background File...")
		precalibration.write(pathClean)

		return precalibration

	@staticmethod
	def reversePrecalibration(precalibration, pathBackground):

		# Reverse the calibration
		logConsole("Reversing the Pre-Calibration...")
		background = PhotomStep.call(precalibration, inverse=True, source_type="EXTENDED")
		background = BarShadowStep.call(background, inverse=True, source_type="EXTENDED")
		background = PathLossStep.call(background, inverse=True, source_type="EXTENDED")
		background = FlatFieldStep.call(background, inverse=True)

		# The result is somehow inverted?
		background.write(pathBackground)

		return background

	@staticmethod
	def subtractBackground(raw, background, pathBNBG):

		# This then removes slit by slit the background
		# TODO : check how the error is propagated
		result = nirspec_utils.apply_master_background(raw, background)

		result.write(pathBNBG)
		logConsole(f"Saving File {pathBNBG}")

		return result



	@staticmethod
	def modelBackgroundFromSlit(slit, precalSlit, radius=4, crop=3, n=0.1):
		"""
		Creates a 2D image model based on the pre-calibration wavelengths positions of the background

		Parameters
		----------
		slit : a slit object, It is assumed that it contains WCS data and that a resampling step has been applied
		precalSlit : a slit object, the same slit just before the resampling step
		radius : float, radius of mask around source
		crop : int, number of lines to ignore above and below
		n : float, fraction of total datapoints to be knots

		Returns
		-------
		np.ndarray, 2D array of a smooth model of background
		"""
		Y, X = np.indices(slit.data.shape)
		_, _, dataLambda = slit.meta.wcs.transform("detector", "world", X, Y)

		Y, X = np.indices(precalSlit.data.shape)
		_, _, precalLambda = precalSlit.meta.wcs.transform("detector", "world", X, Y)

		source = getSourcePosition(slit)

		return BetterBackgroundStep.modelBackgroundFromImage(precalLambda,
										slit.data.copy(),
										slit.err.copy(),
										dataLambda,
										source=source,
										radius=radius,
										crop=crop,
										n=n)

	@staticmethod
	def modelBackgroundFromImage(preCalibrationWavelength : np.ndarray,
								 data : np.ndarray,
								 error : np.ndarray,
								 wavelength : np.ndarray,
								 source = None,
								 radius = 4,
								 crop = 3,
								 n = 0.1,
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
		radius : float, radius of mask around source
		crop : int, number of lines to ignore above and below
		n : float, fraction of total datapoints to be knots
		modelImage : np.ndarray, 2D array representing the envelope-less and noiseless image. Used for testing

		Returns
		-------
		np.ndarray, 2D array of a smooth model of background, or zeros if a fit cannot be made
		"""
		isModelValid = verifySimilarImages(data, modelImage)
		yModel = None

		mask = BetterBackgroundStep.cleanupImage(data, source=source, radius=radius, crop=crop)
		if np.all(mask):
			logConsole("No data was kept in slit. Returning zeros","WARNING")
			return np.zeros_like(preCalibrationWavelength)

		x = BetterBackgroundStep.extractWithMask(wavelength, mask)
		y = BetterBackgroundStep.extractWithMask(data, mask)
		w = 1/np.sqrt(BetterBackgroundStep.extractWithMask(error**2, mask))

		nanMask = (np.isnan(y)) | (np.isnan(w)) | (np.isnan(x))

		if isModelValid:
			yModel = BetterBackgroundStep.extractWithMask(modelImage, mask)
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

		interp = BetterBackgroundStep.makeInterpolation(x,y,w,n)
		# The 2D background model obtained from the 1D spectrum
		return interp(preCalibrationWavelength)

	@staticmethod
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

	@staticmethod
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

	def cleanupImage(self, data : np.ndarray, crop=3, source=None, radius=5):
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