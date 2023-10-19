import io
import time
import logging
from datetime import datetime

import numpy as np
# from torch.utils.tensorboard import SummaryWriter

LOGGER_NAME = 'root'
LOGGER_DATEFMT = '%Y-%m-%d %H:%M:%S'

handler = logging.StreamHandler()

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def add_logging(logs_path, prefix):
    log_name = prefix + datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S') + '.log'
    stdout_log_path = logs_path / log_name

    fh = logging.FileHandler(str(stdout_log_path))
    formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
                                  datefmt=LOGGER_DATEFMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None, mininterval=5):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
 
    def flush(self):
        if len(self.buf) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = time.time()


# class SummaryWriterAvg(SummaryWriter):
#     def __init__(self, *args, dump_period=20, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._dump_period = dump_period
#         self._avg_scalars = dict()

#     def add_scalar(self, tag, value, global_step=None, disable_avg=False):
#         if disable_avg or isinstance(value, (tuple, list, dict)):
#             super().add_scalar(tag, np.array(value), global_step=global_step)
#         else:
#             if tag not in self._avg_scalars:
#                 self._avg_scalars[tag] = ScalarAccumulator(self._dump_period)
#             avg_scalar = self._avg_scalars[tag]
#             avg_scalar.add(value)

#             if avg_scalar.is_full():
#                 super().add_scalar(tag, avg_scalar.value,
#                                    global_step=global_step)
#                 avg_scalar.reset()


class ScalarAccumulator(object):
    def __init__(self, period):
        self.sum = 0
        self.cnt = 0
        self.period = period

    def add(self, value):
        self.sum += value
        self.cnt += 1

    @property
    def value(self):
        if self.cnt > 0:
            return self.sum / self.cnt
        else:
            return 0

    def reset(self):
        self.cnt = 0
        self.sum = 0

    def is_full(self):
        return self.cnt >= self.period

    def __len__(self):
        return self.cnt
