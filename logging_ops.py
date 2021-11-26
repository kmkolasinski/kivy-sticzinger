import time
from typing import Any, Dict, Optional
from kivy.logger import Logger

class measuretime:
    """
    Usage:
        with measuretime("my-name"):
            func1()
            func2()
    """
    def __init__(self, name: str, extra: Optional[Dict[str, Any]] = None, log: bool = True):
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
            Logger.info(
                f"{self.name}: took {self.t:5.3f} [s] {self.params}"
            )
