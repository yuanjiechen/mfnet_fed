import logging.config
import logging
from datetime import datetime

def get_log(name=""):
    logging.config.fileConfig(
        fname="./util/logconfig.ini", 
        defaults={
            "logfilename_s": "log/{:%Y-%m-%d}.log".format(datetime.now()), 
            "logfilename_c": "log/{:%Y-%m-%d}client.log".format(datetime.now())
        }
    )

    logger = logging.getLogger(name)
    # print(name)
    # print(logger.handlers)
    return logger