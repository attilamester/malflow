import inspect
import logging
import os
import traceback
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Union

from util.misc import ensure_dir


class LogLevel(Enum):
    NONE = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4  # Exception


class Logger:
    LOGGER_INSTANCES = {}

    DEFAULT_LOGGER_NAME = "default_logger"
    DEFAULT_LOGGER = None

    @staticmethod
    def datetime_string():
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    @staticmethod
    def _get_stackframe(stack_count: int):
        stack_frames = inspect.stack()[3:3 + stack_count]
        info = []
        for f in stack_frames:
            info.append({"file": f"{f.filename}::{f.lineno}", "func": f.function})
        return info

    @staticmethod
    def __log(logger_type: LogLevel, message):
        if not Logger.DEFAULT_LOGGER:
            Logger.DEFAULT_LOGGER = Logger.get_logger_instance(Logger.DEFAULT_LOGGER_NAME)

        if logger_type == LogLevel.INFO:
            Logger.DEFAULT_LOGGER.info(message)
        elif logger_type == LogLevel.WARNING:
            Logger.DEFAULT_LOGGER.warning(message)
        elif logger_type == LogLevel.ERROR:
            Logger.DEFAULT_LOGGER.error(message)
        elif logger_type == LogLevel.CRITICAL:
            Logger.DEFAULT_LOGGER.critical(message)
        else:
            Logger.DEFAULT_LOGGER.error(message)

    @staticmethod
    def info(message: Union[str, dict, float]):
        Logger.__log(LogLevel.INFO, message)

    @staticmethod
    def warning(message: Union[str, dict, float]):
        Logger.__log(LogLevel.WARNING, message)

    @staticmethod
    def error(message: Union[str, dict, float]):
        Logger.__log(LogLevel.ERROR, message)

    @staticmethod
    def exception(message: Union[str, dict, float, Exception], source=None, include_traceback=True):

        log = {}
        if include_traceback:
            log["traceback"] = traceback.format_exc()

        if source and isinstance(source, str):
            log["source"] = source

        if isinstance(message, str):
            log["message"] = message
        elif isinstance(message, dict):
            log.update(message)
        elif isinstance(message, Exception):
            log["message"] = "".join(traceback.format_exception(
                etype=type(message),
                value=message,
                tb=message.__traceback__
            ))
        else:
            try:
                log["message"] = str(message)
            except:
                pass

        Logger.__log(LogLevel.CRITICAL, log)

    @staticmethod
    def get_logger(logger_name: str) -> logging.Logger:
        logger = logging.getLogger(logger_name)
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-8.8s] %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        Logger.LOGGER_INSTANCES[logger_name] = logger

        return logger

    @staticmethod
    def get_logger_instance(logger_name) -> logging.Logger:
        if logger_name in Logger.LOGGER_INSTANCES:
            return Logger.LOGGER_INSTANCES.get(logger_name)

        return Logger.get_logger(logger_name)

    @staticmethod
    def set_file_logging(file_path):
        if not Logger.DEFAULT_LOGGER:
            ensure_dir(os.path.dirname(file_path))
            if not Logger.DEFAULT_LOGGER:
                Logger.DEFAULT_LOGGER = Logger.get_logger_instance(Logger.DEFAULT_LOGGER_NAME)

        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-8.8s] %(message)s")
        file_handler = RotatingFileHandler(file_path + (".log" if not file_path.endswith(".log") else ""))
        file_handler.setFormatter(log_formatter)
        Logger.DEFAULT_LOGGER.addHandler(file_handler)
