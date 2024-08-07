import os

os.environ['CRDS_PATH'] = '/net/CLUSTER/VAULT/users/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
import stdatamodels.jwst.datamodels as dm
from jwst.wavecorr import WavecorrStep
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.pixel_replace import PixelReplaceStep
from jwst.resample import ResampleSpecStep
from jwst.extract_1d import Extract1dStep

from utils import logConsole
import BetterBackgroundSubtractStep as BkgSubtractStep


def Stage1(uncal,path):
	logConsole(f"Starting Stage 1 on {uncal}")
	if os.path.exists(uncal.replace("_uncal", "_rate")):
		return
	# Steps Params By Pablo Arrabal Haro
	"""steps = {
			'jump': {'expand_large_events': True,
					 # 1st flag groups after jump above DN threshold.
					 'after_jump_flag_dn1': 0,
					 # 1st flag groups after jump groups within
					 # specified time.
					 'after_jump_flag_time1': 0,
					 # 2nd flag groups after jump above DN threshold.
					 'after_jump_flag_dn2': 0,
					 # 2nd flag groups after jump groups within
					 # specified time.
					 'after_jump_flag_time2': 0,
					 # Minimum required area for the central saturation
					 # of snowballs.
					 'min_sat_area': 15.0,
					 # Minimum area to trigger large events processing.
					 'min_jump_area': 15.0,
					 # The expansion factor for the enclosing circles
					 # or ellipses.
					 'expand_factor': 2.0}
			}"""
	det1 = Detector1Pipeline()
	det1.save_results = True
	det1.output_dir = path
	det1.run(uncal)
	del det1
	return


def Stage2(rate,path):
	logConsole(f"Starting Stage 2")
	if not os.path.exists(rate.replace("rate", "srctype")):
		steps = {'srctype': {'save_results': True},
				 'photom': {'skip': True},
				 'flat_field': {'skip': True},
				 'master_background_mos': {'skip': True},
				 'wavecorr': {'skip': True},
				 'pathloss': {'skip': True},
				 'barshadow': {'skip': True},
				 'pixel_replace': {'skip': True},
				 'extract_1d': {'skip': True},
				 'cube_build': {'skip': True},
				 'resample_spec': {'skip': True}}

		spec2 = Spec2Pipeline(steps=steps)
		spec2.output_dir = path
		spec2.run(rate)
		del spec2

	if not os.path.exists(rate.replace("rate", "bkg")):
		BkgSubtractStep.BetterBackgroundStep(rate.replace("_rate", "_srctype"))

	bkg = rate.replace("_rate", "_bkg")

	if not os.path.exists(bkg.replace("_bkg", "_bkg_photomstep")):
		logConsole("Restarting Pipeline Stage 2")
		# Steps :
		# wavecorr
		# flat field
		# path loss
		# bar shadow
		# photom
		# pixel replace
		# rectified 2D -> Save
		# spectral extraction -> Save

		with dm.open(bkg) as data:
			logConsole("Successfully loaded _bkg file")
			calibrated = WavecorrStep.call(data)
			calibrated = FlatFieldStep.call(calibrated)
			calibrated = PathLossStep.call(calibrated)
			calibrated = BarShadowStep.call(calibrated)
			calibrated = PhotomStep.call(calibrated, output_dir=path, save_results=True)
			calibrated = PixelReplaceStep.call(calibrated)
			calibrated = ResampleSpecStep.call(calibrated, output_dir=path, save_results=True)
			calibrated = Extract1dStep.call(calibrated, output_dir=path, save_results=True)
			del calibrated
