import os
import time

from ..utils.utils import logConsole, getSourcePosition, PathManager, getCRDSPath

os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from scipy.ndimage import generic_filter
from scipy.stats import median_abs_deviation
import numpy as np
from stdatamodels.jwst.datamodels import MultiSlitModel, SlitModel
import pandas as pd

from ..Pipeline.BSplineLSQ import BSplineLSQ


class BetterBackgroundStep:
	def __init__(self,
				 ratePath,
				 s2d,
				 cal,
				 radius=4,
				 crop=3,
				 interpolationKnots=0.15,
				 curvatureConstraint=1,
				 endpointConstraint=0.1,
				 kernelSize=(1,15),
				 Nsigma=10):
		"""
		Creates a _bkg-BNBG.fits file from a _s2d file and _cal file.

		Parameters
		----------
		ratePath : PathManager
		    Path based on the initial rate file created at the start of Stage 2.

		s2d : MultiSlitModel
		    DataModel after Stage 2. Must be a _s2d file.

		cal : MultiSlitModel
		    DataModel after Stage 2. Must be a _cal file.

		radius : float
		    The radius of extraction around the source.

		crop : int
		    Number of lines ignored at the top and bottom of the resampled 2D image.

		interpolationKnots : float
		    Fraction of total data points used as knots (between 0 and 1).

		curvatureConstraint : float
		    A hyperparameter that controls how much the curvature is minimized.
		    Useful when gaps are present in the data. If set to 0, curvature is ignored.

		endpointConstraint : float
		    A hyperparameter that controls the smoothing at the endpoints of the spline.
		    If set to 0, endpoint slopes are ignored.

		kernelSize : tuple[int, int]
		    Size of the kernel used for median filtering (e.g., (1,15)).

		Nsigma : float
		    Number of standard deviations above which pixels in error and data
		    will be masked after median filtering.
		"""

		logConsole("Initializing Better Background NIRSpec")

		self.ratePath : PathManager = ratePath
		self.s2d : MultiSlitModel = s2d
		self.cal : MultiSlitModel = cal
		self.radius : float = radius
		self.crop : int = crop
		self.interpolationKnots : float = interpolationKnots
		self.curvatureConstraint : float = curvatureConstraint
		self.endpointConstraint : float = endpointConstraint
		self.kernelSize : tuple = kernelSize
		self.Nsigma : float = Nsigma

		# Inits for later calls
		self.background = None

	def __call__(self):
		return self.process()

	def process(self):
		"""
		Core method of BNBG.
		Calculates / gets a MultiSlitModel background file (bkg-BNBG).
		Subtracts it to a similar MultiSlitModel (cal) to generate a background corrected MultiSlitModel (cal-BNBG).

		Returns
		-------
		results : MultiSlitModel
			a copy of ``cal``, for each slit,
			slit.data and slit.err correspond to the modelled background, and it's error.
			This is the 'pure' spectral background, with no spatial effects.

		"""
		logConsole("Calculating background")
		self.background = self.ratePath.openSuffix("bkg-BNBG", self._generateBackground)

		# Subtract background from original
		logConsole("Subtracting Background")
		return self.ratePath.openSuffix("cal-BNBG",
										lambda: self._subtractBackground)



	def _generateBackground(self):
		"""
		Calculates background per slit for a MultiSlitModel (s2d)

		Returns
		-------
		bkg : MultiSlitModel
			Background MultiSlitModel. 1 Shutter slits will be skipped.
		"""

		# Will serve as a logger file, giving info on the fit of each slit
		fitInfo = pd.DataFrame(
			{"slit": [], "source": [], "datapoints": [], "insideKnots": [], "chi2": [], "reducedChi2": [], "dof": []})

		# Make copy or else will be overwritten
		cal = self.cal.copy()
		for i, slit in enumerate(self.s2d.slits):
			logConsole(f"Calculating Background for slit {slit.name}")

			s2dSlit = slit
			calSlit = cal.slits[i]

			if len(slit.shutter_state) == 1:
				logConsole(f"Only 1 shutter in slit, skipping...", "WARNING")
				calSlit.data = np.zeros_like(calSlit.data)
				# calSlit.err remains unchanged
				fitInfo.loc[len(fitInfo)] = [slit.name,
											 slit.source_id,
											 None,
											 None,
											 None,
											 None,
											 None]
				continue

			bspline = self._modelBackgroundFromSlit(s2dSlit)

			if bspline is None:
				calSlit.data = np.zeros_like(calSlit.data)
				# calSlit.err remains unchanged
				fitInfo.loc[len(fitInfo)] = [slit.name,
											 slit.source_id,
											 None,
											 None,
											 None,
											 None,
											 None]
			else:
				Y, X = np.indices(calSlit.data.shape)
				_, _, targetLambda = calSlit.meta.wcs.transform("detector", "world", X, Y)

				calSlit.data, calSlit.err = bspline(targetLambda), bspline.getError(targetLambda)
				fitInfo.loc[len(fitInfo)] = [slit.name,
											 slit.source_id,
											 len(bspline.x),
											 bspline.nInsideKnots,
											 bspline.getChiSquare(),
											 bspline.getReducedChi(),
											 bspline.getDegreesOfFreedom()]

			logConsole("Background Calculated!")

		logConsole("Saving Clean Background File...")

		pathClean = self.ratePath.withSuffix("bkg-BNBG")
		cal.save(pathClean)
		fitInfo.to_csv(pathClean.replace("fits", "csv"), index=False, sep="\t")

		return cal

	def _modelBackgroundFromSlit(self, s2dSlit : SlitModel):

		Y, X = np.indices(s2dSlit.data.shape)
		_, _, dataLambda = s2dSlit.meta.wcs.transform("detector", "world", X, Y)

		source = getSourcePosition(s2dSlit)

		bspline = self._modelBackgroundFromImage(s2dSlit.data.copy(),
												 s2dSlit.err.copy(),
												 dataLambda,
												 source=source)
		return bspline


	def _modelBackgroundFromImage(self,
								  data : np.ndarray,
								  error : np.ndarray,
								  wavelength : np.ndarray,
								  source : float | None = None):
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

		source : float,
			Vertical position of the source in pixel space.

		Returns
		-------
		results : BSplineLSQ
			bspline object fitted on the data.
		"""

		# Getting 1D arrays
		x,y,dy = self._getDataWithMask(data.copy(), error.copy(), wavelength.copy(), source=source)

		# Check if enough data points (spline of order 4 needs at least 10 points)
		if (x is None and y is None and dy is None) or len(x) < 10:
			logConsole("Not enough points to fit. Returning zeros", "WARNING")
			return None

		# Weights
		w = 1/dy**2

		# Creating bspline object
		logConsole(f"Starting BSpline fitting with {len(x)} data points (at least {self.interpolationKnots * len(x)} inside knots)...")
		startTime = time.time()

		bspline = BSplineLSQ(x,
							 y,
							 w,
							 interpolationKnots=self.interpolationKnots,
							 curvatureConstraint=self.curvatureConstraint,
							 endpointConstraint=self.endpointConstraint)

		logConsole(f"Finished fitting in {round(time.time() - startTime,3)}s")

		logConsole(f"Model found with reduced_chi2 = {np.round(bspline.getReducedChi(),5)}")

		return bspline

	def _getDataWithMask(self,
						 data : np.ndarray,
						 error : np.ndarray,
						 wavelength : np.ndarray,
						 source : float | None = None):
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

		Returns
		-------
		x, y, dy : (np.ndarray, np.ndarray, np.ndarray)
			1D arrays, respectively wavelength, flux and error
		"""

		data = data
		wavelength = wavelength
		error = error

		mask = self._cleanupImage(data.copy(), error.copy(), source=source)
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

	def _cleanupImage(self,
					  data : np.ndarray,
					  error : np.ndarray,
					  source : float | None = None):
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

		source : float
			The vertical position, in number of lines, from which to mask the source. If None, no source will be masked

		Returns
		-------
		mask : np.ndarray
			Array of boolean where False means the pixel was kept for background subtraction and True means it was rejected
		"""
		data = data
		error = error

		Y, X = np.indices(data.shape)
		# Gets rid of negative values, crops the top and bottom of the image, ignores pixels marked as nan
		mask = (data <= 0) | (Y < self.crop) | (Y > Y.max() - self.crop) | np.isnan(data)

		# If source in the image, remove lines in a radius around
		# This also works if the source is not in frame and the returned position is 1e48
		if source is not None:
			# TODO ? : Case of secondary source?
			mask = mask | (np.round(abs(Y - source)) < self.radius)

		data[mask] = np.nan
		error[mask] = np.nan
		medianData = generic_filter(data, lambda x : np.nanmedian(x), size=self.kernelSize, mode="nearest")
		medianError = generic_filter(error, lambda x : np.nanmedian(x), size=self.kernelSize, mode="nearest")

		medianSubtractData = data - medianData
		_ = medianSubtractData.ravel()
		_ = _[np.isfinite(_)]
		MADData = median_abs_deviation(_)

		medianSubtractError = error - medianError
		_ = medianSubtractError.ravel()
		_ = _[np.isfinite(_)]
		MADError = median_abs_deviation(_)

		mask = (mask
				| (np.abs(medianSubtractData) > self.Nsigma*MADData)
				| (np.abs(medianSubtractError) > self.Nsigma*MADError))

		return mask

	def _subtractBackground(self):
		"""
		Subtracts the background data from raw data.
		Also adds the errors appropriately.

		Returns
		-------
		result : MultiSlitModel
			The subtracted background

		"""
		result = self.cal.copy()
		for i, slit in enumerate(result.slits):
			rawSlit = slit
			bkgSlit = self.background.slits[i]

			slit.data = rawSlit.data - bkgSlit.data
			slit.err = np.sqrt((rawSlit.err ** 2 + bkgSlit.err ** 2)) # The error will be larger or equal
			# The original master background from the pipeline does not propagate errors
			# This is unfair imo

		pathBNBG = self.ratePath.withSuffix("cal-BNBG")
		result.save(pathBNBG)
		logConsole(f"Saving File {os.path.basename(pathBNBG)}")

		return result

##########
# STATIC
##########

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