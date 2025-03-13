import os
import logging
import threading

from .CrdsSetup import getCRDSPath

os.environ['CRDS_PATH'] = getCRDSPath()
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

_HAS_LOADED = False

# Get logger of jwst pipeline
import jwst.step # Only serves to initialize the jwst logger
logger = logging.getLogger("stpipe")

def _setup():
	global _HAS_LOADED
	_HAS_LOADED = False

	if logger.hasHandlers():
		print("Logger loaded")
		_HAS_LOADED = True
		formatter = logger.handlers[0].formatter
		logFormat = formatter._fmt
		newFormat = logFormat.replace("%(message)s", "[%(native_id)d] %(message)s")
		logger.handlers[0].setFormatter(logging.Formatter(newFormat))

	# Add 'native_id' to LogRecord dynamically
	oldFactory = logging.getLogRecordFactory()

	def _record_factory(*args, **kwargs):
		record = oldFactory(*args, **kwargs)
		record.native_id = threading.get_native_id()  # Add native thread ID
		return record

	logging.setLogRecordFactory(_record_factory)

_setup()

def logConsole(text : str, source=None):
	"""
	 Logger : displays time + log

	Parameters
	----------
	text : str
		Text to display
	source : str
		 WARNING, ERROR, DEBUG, INFO/None
	"""
	if not _HAS_LOADED:
		raise ImportError("Couldn't load the JWST logger")

	text = f" [BetterBackground] : {text}"
	logType = {"WARNING": lambda : logger.warning(text),
			   "ERROR": lambda : logger.error(text),
			   "DEBUG": lambda : logger.debug(text)
			   }

	logType.get(source,lambda : logger.info(text))()