from __future__ import annotations

import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    _P = ParamSpec("_P")
    _R = TypeVar("_R")
    _RD = TypeVar("_RD", bound=Mapping[Any, Any])


def timed(f: Callable[_P, _R]) -> Callable[_P, _R]:
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
    def wrap(*args: _P.args, **kw: _P.kwargs) -> _R:
        ts = datetime.datetime.now(tz=datetime.UTC)
        try:
            result = f(*args, **kw)
        finally:
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
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


@overload
def npy_cached(
    path: Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    ...


def npy_cached(
    path: Path | None | Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
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

    def _npy_cached(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(f)
        def wrap(*args: _P.args, **kw: _P.kwargs) -> _R:
            cache_path = path(*args, **kw) if callable(path) else path
            if cache_path is None:
                return f(*args, **kw)
            try:
                arr: _R = np.load(cache_path, allow_pickle=load_pickle)[()]
            except FileNotFoundError:
                arr = f(*args, **kw)
                # Saving pickeld
                np.save(cache_path, np.asanyarray(arr), allow_pickle=save_pickle)

            return arr

        return wrap  # type: ignore[return-value]

    return _npy_cached


@overload
def npy_cached_dict(
    path: Path | None,
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _RD]], Callable[_P, _RD]]:
    ...


@overload
def npy_cached_dict(
    path: Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _RD]], Callable[_P, _RD]]:
    ...


def npy_cached_dict(
    path: Path | None | Callable[_P, Path | None],
    *,
    load_pickle: bool = False,
    save_pickle: bool = True,
) -> Callable[[Callable[_P, _RD]], Callable[_P, _RD]]:
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

    def _npy_cached(f: Callable[_P, _RD]) -> Callable[_P, _RD]:
        @wraps(f)
        def wrap(*args: _P.args, **kw: _P.kwargs) -> _RD:
            cache_path = path(*args, **kw) if callable(path) else path
            if cache_path is None:
                return f(*args, **kw)
            cache_path = cache_path.with_suffix(".npz")
            try:
                data = np.load(cache_path, allow_pickle=load_pickle)
                obj: _RD = {f: data[f][()] for f in data.files}  # type: ignore not _RD
            except FileNotFoundError:
                obj = f(*args, **kw)
                np.savez(cache_path, **obj)

            return obj

        return wrap  # type: ignore[return-value]

    return _npy_cached
