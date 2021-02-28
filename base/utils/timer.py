# timer.py
# from https://github.com/realpython/codetiming

from contextlib import ContextDecorator
import time
from typing import Any, Dict


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""
    timers: Dict[str, float] = dict()  # if a name is given, the time is kept in the global class type
    instance_num: Dict[str, int] = dict()  # used to compute running average
    averages: Dict[str, float] = dict()

    def __init__(self, name=None, text="Elapsed time: {:0.4f} seconds", logger=print):
        r"""
        :param name: if given, total and average time is stored in class variable
        :param text: text to print when timer is stopped
        :param logger: function to log text, set to falsy not to log
        """
        self.name, self.text, self.logger = name, text, logger
        if self.name:
            self.timers.setdefault(self.name, 0)
            self.instance_num.setdefault(self.name, 0)
            self.averages.setdefault(self.name, 0)
        self._start_time = None

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time
            self.instance_num[self.name] += 1
            self.averages[self.name] += (self.averages[self.name] + (elapsed_time - self.averages[self.name]) /
                                         (self.instance_num[self.name] + 1))
        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
