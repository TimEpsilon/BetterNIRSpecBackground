from astropy.io import fits
from scipy.signal import find_peaks_cwt
from utils import *
from scipy.optimize import curve_fit as cfit

def BetterBackgroundStep(name,threshold=0.7):
	"""
	 Creates a _bkg file from a _srctype file

	 Params
	 ---------
	 name : str
		Path to the file to open. Must be a _srctype
	 threshold : float
		A number between 0 and 1. Represents the proportion of high intensity pixels that will be cutoff,
		1 being no pixels removed, 0 being every pixel removed
	"""
	p = 2
	if not "_srctype" in name:
		logConsole(f"{name.split('/')[-1]} not a _srctype file. Skipping...",source="WARNING")
		pass

	# 1st draft Algorithm :
	logConsole(f"Starting Custom Background Substraction on {name.split('/')[-1]}",source="BetterBackground")
	multi_hdu = fits.open(name)

	# For a given _srctype, for every SCI inside
	for i,hdu in enumerate(multi_hdu):
		if not hdu.name == 'SCI':
			continue
		hdr = hdu.header
		data = np.ma.masked_invalid(hdu.data)

		#TODO : Eventually, work on error propagation

		shutter_id = WhichShutterOpen(hdr)
		if shutter_id is None:
			continue

		logConsole(f"Extension {i} is SCI. Open shutter is {shutter_id+1}",source="BetterBackground")

		slice_indices = SelectSlice(data)

		if np.any(slice_indices is None):
			logConsole("Can't find 3 spectra. Defaulting to equal slices",source="WARNING")
			n = data.shape[0]
			xmin = np.array([0,int(n/3),int(2*n/3)])
			xmax = np.array([int(n/3),int(2*n/3),n])
			slice_indices = np.array([xmin,xmax]).T

		bkg_slice = []
		bkg_interp = []
		for j in range(2):
			# Get 2 background strips
			bkg_slice.append(data[slice_indices[shutter_id-j-1][0]:slice_indices[shutter_id-j-1][1],:])
			_ = bkg_slice[j] < 0
			bkg_slice[j][_] = np.nan
			bkg_slice[j][_].mask = True

			bkg_interp.append(SubtractSignalToBackground(bkg_slice[j], threshold, power=p))


		# Remove pixels + interpolate on a given strip (ignore source strip)
		new_bkg = np.copy(data)
		new_bkg[:,:] = np.nan

		new_bkg[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1],:] = bkg_interp[0]
		new_bkg[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1],:] = bkg_interp[1]

		non_nan = np.where(np.logical_not(np.isnan(new_bkg)))

		x = non_nan[0]
		y = non_nan[1]
		z = new_bkg[non_nan]

		interp = IDWExtrapolation(np.c_[x, y], z, power=p)

		X = np.arange(new_bkg.shape[0])
		Y = np.arange(new_bkg.shape[1])
		YY,XX = np.meshgrid(Y,X)
		new_bkg = interp(XX,YY)

		hdu.data = np.ma.getdata(data - new_bkg)
		

	logConsole(f"Saving File {name.split('/')[-1]}",source="BetterBackground")
	multi_hdu.writeto(name.replace("_srctype","_bkg"),overwrite=True)
	multi_hdu.close()
	pass


def SubtractSignalToBackground(bkg, threshold, selectionMethod="median", interpMethod="IDW", **Kwargs):
	"""
	Subtracts the high value signal from the background and interpolates those flagged pixels

	Params
	-----------
	bkg : 2D array
		The background strip

	threshold : float

	selectionMethod : str
		Either "median" or "minmax".
		"median" means that the selection uses the q-th quantile, q the threshold.
		If threshold = 0.3, we keep 30% of the lowest values. This is useful if we want a set amount of pixels
		"minmax" means that the selection is based on the range between the min and max value.
		If threshold = 0.3, we keep all values below 30% of the range. This is useful if we want a max pixel value.

	interpMethod : str
		Either "IDW", "NN"
		"IDW" is Inverse Distance Weighting, a weighted average interpolation method based on the distance to known points.
			Additional arguments are "power"
		"NN" is Nearest Neighbour, which assigns to each unknown point the value of the nearest known point. Unpractical

	**Kwargs : additional arguments for the interpolation. Careful to use the appropriated keywords
	"""
	bkg = np.ma.masked_invalid(bkg)
	# Determine non background sources : sudden spikes, high correlation with source strip, etc -> flag pixels
	if selectionMethod == "median":
		mask = bkg < np.nanquantile(bkg.compressed(), threshold)
	elif selectionMethod == "minmax":
		mask = bkg < bkg.min() + (bkg.max() - bkg.min()) * threshold
	else :
		logConsole(f"Unknown selectionMethod {selectionMethod}, defaulting to median",source="WARNING")
		mask = bkg < np.nanquantile(bkg.compressed(), threshold)

	mask = np.ma.getdata(mask)

	logConsole(f"Ratio of kept pixels is {round(mask.sum()/len(bkg.ravel()),3)}")

	master_background = np.ma.array(bkg,mask=~mask,fill_value=np.nan)

	non_nan = np.where(mask)
	x = non_nan[0]
	y = non_nan[1]
	z = master_background[non_nan]
	z = np.ma.getdata(z)

	if interpMethod == "NN":
		interp = NNExtrapolation(np.c_[x, y], z)
	else:
		interp = IDWExtrapolation(np.c_[x, y], z, **Kwargs)

	X = np.arange(bkg.shape[0])
	Y = np.arange(bkg.shape[1])
	YY,XX = np.meshgrid(Y,X)

	return interp(XX,YY)


def SelectSlice(data):
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross section, finding a pixel position for each peak,
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	data : 2D array
		An individual image obtained from an individual "SCI" extension in a fits

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit

	"""
	# Get vertical cross section by summing horizontally
	horiz_sum = np.mean(data,axis=1)

	# Determine 3 maxima for 3 slits
	peaks = []
	j = 2
	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1
	if not len(peaks) == 3:
		return None
	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(range(len(horiz_sum)),horiz_sum,peaks))

	if np.any(peaks > len(horiz_sum)) or np.any(peaks < 0):
		return None

	# Cut horizontally at midpoint between maxima -> 3 strips
	slice_indices = getPeakSlice(peaks,0,len(horiz_sum))

	return slice_indices


def getPeaksPrecise(x,y,peaks):
	"""
	Returns the sub-pixel peaks
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5],full_output=True)
	except :
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return peaks
	return np.array(coeff[:3])


def getPeakSlice(peaks,imin,imax):
	"""
	Returns slices for a set of peaks
	"""
	d1 = (peaks[1] - peaks[0])/2
	d2 = (peaks[2] - peaks[1])/2

	xmin = np.array([round(max(imin,peaks[0]-d1)),round(peaks[1]-d1),round(peaks[2]-d2)])
	xmax = np.array([round(peaks[0]+d1), round(peaks[1]+d2), round(min(imax,peaks[2]+d2))])

	return np.array([xmin,xmax]).T


#BetterBackgroundStep("jw01345063001_03101_00001_nrs1_srctype.fits")