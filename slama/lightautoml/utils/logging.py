"""Utils for logging."""

import io
import logging
import os
import sys

from .. import _logger


formatter_debug = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(pathname)s.%(funcName)s:%(lineno)d\t%(message)s")
formatter_default = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")

INFO2 = 17
INFO3 = 13


def add_logging_level(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    assert (levelNum > 0) and (levelNum < 50)
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


add_logging_level("INFO2", INFO2)
add_logging_level("INFO3", INFO3)


class LoggerStream(io.IOBase):
    def __init__(self, logger, verbose_eval=100) -> None:
        super().__init__()
        self.logger = logger
        self.verbose_eval = verbose_eval
        self.counter = 1

    def write(self, message):
        if message == "\n":
            return
        iter_num = message.split("\t")[0]
        if (iter_num == "[1]") or (iter_num == "0:") or ((iter_num[-1] != "]") and (iter_num[-1] != ":")):
            self.logger.info3(message.rstrip())
            return

        if self.counter < self.verbose_eval - 1:
            self.logger.debug(message.rstrip())
            self.counter += 1
        else:
            self.logger.info3(message.rstrip())
            self.counter = 0


def verbosity_to_loglevel(verbosity: int):
    if verbosity <= 0:
        log_level = logging.ERROR
    elif verbosity == 1:
        log_level = logging.INFO
    elif verbosity == 2:
        log_level = logging.INFO2
    elif verbosity == 3:
        log_level = logging.INFO3
    else:
        log_level = logging.DEBUG

    return log_level


def get_stdout_level():
    for handler in _logger.handlers:
        if type(handler) == logging.StreamHandler:
            return handler.level
    return _logger.getEffectiveLevel()


def set_stdout_level(level):
    _logger.setLevel(logging.DEBUG)

    has_console_handler = False

    for handler in _logger.handlers:
        if type(handler) == logging.StreamHandler:
            if handler.level == level:
                has_console_handler = True
            else:
                _logger.handlers.remove(handler)

    if not has_console_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter_default)
        handler.setLevel(level)

        _logger.addHandler(handler)


def add_filehandler(filename: str, level=logging.DEBUG):
    if filename:
        has_file_handler = False

        for handler in _logger.handlers:
            if type(handler) == logging.FileHandler:
                if handler.baseFilename == filename or handler.baseFilename == os.path.join(os.getcwd(), filename):
                    has_file_handler = True
                else:
                    _logger.handlers.remove(handler)

        if not has_file_handler:
            file_handler = logging.FileHandler(filename, mode="w")

            if level == logging.DEBUG:
                file_handler.setFormatter(formatter_debug)
            else:
                file_handler.setFormatter(formatter_default)

            file_handler.setLevel(level)

            # if handler_filter:
            #     file_handler.addFilter(handler_filter)

            _logger.addHandler(file_handler)
    else:
        for handler in _logger.handlers:
            if type(handler) == logging.FileHandler:
                _logger.handlers.remove(handler)


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv
