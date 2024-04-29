import os

os.environ['CRDS_PATH'] = '/home/tim/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Spec2Pipeline

working_dir = "./detailledPipeline/BasePipeline/"

Spec2Pipeline.call(working_dir+"jw01345070001_05101_00003_nrs1_rate.fits",save_results=True,output_dir=working_dir)