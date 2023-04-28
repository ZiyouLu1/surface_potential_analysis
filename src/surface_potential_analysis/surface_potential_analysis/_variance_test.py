from __future__ import annotations

from typing import Generic, TypeVar

TestType = TypeVar("TestType", bound=int, covariant=True)


class Test(Generic[TestType]):
    pass


_LInv = TypeVar("_LInv", bound=int)
_LCov = TypeVar("_LCov", bound=int, covariant=True)
_LCon = TypeVar("_LCon", bound=int, contravariant=True)


def b(_basis: Test[_LInv]) -> None:
    return


def c(_basis: Test[_LCov]) -> None:
    return


def d(_basis: Test[_LCon]) -> None:
    return


def e(_basis: Test[int]) -> None:
    return


def b_inv(basis: Test[_LInv]) -> None:
    b(basis)


def b_cov(basis: Test[_LCov]) -> None:
    b(basis)


def b_con(basis: Test[_LCon]) -> None:
    b(basis)


def c_inv(basis: Test[_LInv]) -> None:
    c(basis)


def c_cov(basis: Test[_LCov]) -> None:
    c(basis)


def c_con(basis: Test[_LCon]) -> None:
    c(basis)


def d_inv(basis: Test[_LInv]) -> None:
    d(basis)


def d_cov(basis: Test[_LCov]) -> None:
    d(basis)


def d_con(basis: Test[_LCon]) -> None:
    d(basis)


def e_inv(basis: Test[_LInv]) -> None:
    e(basis)


def e_cov(basis: Test[_LCov]) -> None:
    e(basis)


def e_con(basis: Test[_LCon]) -> None:
    e(basis)
