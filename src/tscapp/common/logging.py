import logging

logger = logging.getLogger(__name__)

def configure_logging(*, level: str | int = logging.INFO):
    logging.basicConfig(level=level)
    logger.info("configure logging")
    for name, l in {
        # "library": logging.INFO
    }.items():
        logging.getLogger(name).setLevel(l)