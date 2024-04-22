from utils import logConsole
import os
os.environ['CRDS_PATH'] = '/home/tdewachter/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'
from glob import glob
from jwst.pipeline import Spec2Pipeline

from utils import logConsole

logConsole("test")