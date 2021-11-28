import time
from functools import wraps
from typing import Any, Dict, Optional
from kivy.logger import Logger


class measuretime:
    """
    Usage:
        with measuretime("my-name"):
            func1()
            func2()
    """

    def __init__(
        self, name: str, extra: Optional[Dict[str, Any]] = None, log: bool = True
    ):
        self.name = name
        self.extra = extra
        self.log = log

    @property
    def params(self) -> str:
        if self.extra is None:
            return ""

        params = []
        for k in sorted(self.extra):
            params.append(f"{k}={self.extra[k]}")
        params = ", ".join(params)
        return f"PARAMS: {params}"

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.t = time.perf_counter() - self.t
        if self.log:
            Logger.info(f"{self.name}: took {self.t:5.3f} [s] {self.params}")


class elapsedtime:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.seconds = time.perf_counter() - self.start


def profile(method):
    @wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        with measuretime(f"Calling {self.__class__.__name__}.{method.__name__}"):
            method_output = method(self, *method_args, **method_kwargs)
        return method_output

    return _impl
