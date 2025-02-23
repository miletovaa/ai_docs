import os
import logging
import inspect

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name=None):
    if name is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_name = os.path.splitext(os.path.basename(module.__file__))[0] if module else "default"
        name = script_name

    log_filename = os.path.join(LOG_DIR, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger