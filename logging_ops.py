import time
from functools import wraps
from typing import Any, Dict, Optional
from kivy.logger import Logger
from collections import defaultdict

PROFILER_STATS = defaultdict(list)
PROFILER_HISTORY = []


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
        self.seconds = time.perf_counter() - self.t
        if self.log and self.seconds > 0.001:
            Logger.info(f"{self.name}: took {self.seconds:5.3f} [s] {self.params}")


class elapsedtime:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        self.seconds = time.perf_counter() - self.start


def profile(name: str = None, key=None):
    def _profile(method):
        @wraps(method)
        def _impl(self, *method_args, **method_kwargs):

            if name is None:
                name_key = f"{self.__class__.__name__}.{method.__name__}"
            else:
                name_key = name

            if key is not None:
                key_str = key(self, *method_args, **method_kwargs)
            else:
                key_str = ""

            with measuretime(f"Calling {name_key} {key_str}") as dt:
                method_output = method(self, *method_args, **method_kwargs)

            if dt.seconds > 0.001:
                stats_key = f"{name_key}/{key_str}"

                PROFILER_STATS[stats_key].append(dt.seconds)
                if len(PROFILER_STATS[stats_key]) > 100:
                    PROFILER_STATS[stats_key] = PROFILER_STATS[stats_key][1:]

                PROFILER_HISTORY.append(
                    (float(f"{dt.t:.3f}"), float(f"{dt.seconds:.3f}"), stats_key)
                )

            return method_output

        return _impl

    return _profile


def reset_profiler():
    global PROFILER_STATS, PROFILER_HISTORY
    PROFILER_STATS = defaultdict(list)
    PROFILER_HISTORY = []


def get_profiler_metrics(reset: bool = True):

    metrics = []
    for key, values in PROFILER_STATS.items():
        total_time = sum(values)
        avg_time = total_time / len(values)
        num_samples = len(values)

        metrics.append((key, avg_time, total_time, num_samples))

    metrics = sorted(metrics, key=lambda x: x[1])

    metrics_dict = {"metrics": [], "history": PROFILER_HISTORY}
    for data in metrics:
        metrics_dict["metrics"].append(
            {
                "key": data[0],
                "avg_time": f"{data[1]:.3f}",
                "total_time": f"{data[2]:.3f}",
                "count": data[3],
            }
        )

    if reset:
        reset_profiler()

    return metrics_dict
