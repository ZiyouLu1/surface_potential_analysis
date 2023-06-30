from __future__ import annotations

import datetime
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    P = ParamSpec("P")
    R = TypeVar("R")


def timed(f: Callable[P, R]) -> Callable[P, R]:
    """
    Log the time taken for f to run.

    Parameters
    ----------
    f : Callable[P, R]
        The function to time

    Returns
    -------
    Callable[P, R]
        The decorated function
    """

    @wraps(f)
    def wrap(*args: P.args, **kw: P.kwargs) -> R:
        ts = datetime.datetime.now(tz=datetime.UTC)
        result = f(*args, **kw)
        te = datetime.datetime.now(tz=datetime.UTC)
        print(f"func: {f.__name__} took: {(te - ts).total_seconds()} sec")  # noqa: T201
        return result

    return wrap  # type: ignore[return-value]


@overload
def npy_cached(
    path: Path | None,
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


@overload
def npy_cached(
    path: Callable[P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


def npy_cached(
    path: Path | None | Callable[P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Cache the response of the function at the given path.

    Parameters
    ----------
    path : Path | Callable[P, Path]
        The file to read.
    load_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files.
        Reasons for disallowing pickles include security, as loading pickled data can execute arbitrary code.
        If pickles are disallowed, loading object arrays will fail. default: False
    save_pickle : bool, optional
        Allow saving pickled objects. default: True

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
    """

    def _npy_cached(f: Callable[P, R]) -> Callable[P, R]:
        @wraps(f)
        def wrap(*args: P.args, **kw: P.kwargs) -> R:
            cache_path = path(*args, **kw) if callable(path) else path
            if cache_path is None:
                return f(*args, **kw)
            try:
                arr: R = np.load(cache_path, allow_pickle=load_pickle)[()]
            except FileNotFoundError:
                arr = f(*args, **kw)
                # Saving pickeld
                np.save(cache_path, arr, allow_pickle=save_pickle)

            return arr

        return wrap  # type: ignore[return-value]

    return _npy_cached
