import datetime
from functools import wraps
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


def timed(f: F) -> F:
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.datetime.now()
        result = f(*args, **kw)
        te = datetime.datetime.now()
        print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")
        return result

    return wrap  # type: ignore
