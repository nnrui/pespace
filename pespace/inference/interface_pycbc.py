import logging

try:
    import pycbc
except ImportError:
    logging.error(
        "pycbc is not installed by default, if sampler interface in pycbc is needed, "
        "please install pycbc manually."
    )
    raise
