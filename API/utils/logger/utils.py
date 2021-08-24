from contextlib import contextmanager
import logging

@contextmanager
def mute_logging():
    import logging
    logging.disable()
    try:
        yield
    finally:
        logging.disable(0)