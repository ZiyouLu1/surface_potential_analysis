from __future__ import annotations

from typing import Generic, TypeVar

TestType_co = TypeVar("TestType_co", bound=int, covariant=True)


class Test(Generic[TestType_co]):
    pass


_LInv = TypeVar("_LInv", bound=int)
_L_co = TypeVar("_L_co", bound=int, covariant=True)
_L_contra = TypeVar("_L_contra", bound=int, contravariant=True)


def b(_basis: Test[_LInv]) -> None:
    return


def c(_basis: Test[_L_co]) -> None:
    return


def d(_basis: Test[_L_contra]) -> None:
    return


def e(_basis: Test[int]) -> None:
    return


def b_inv(basis: Test[_LInv]) -> None:
    b(basis)


def b_cov(basis: Test[_L_co]) -> None:
    b(basis)


def b_con(basis: Test[_L_contra]) -> None:
    b(basis)


def c_inv(basis: Test[_LInv]) -> None:
    c(basis)


def c_cov(basis: Test[_L_co]) -> None:
    c(basis)


def c_con(basis: Test[_L_contra]) -> None:
    c(basis)


def d_inv(basis: Test[_LInv]) -> None:
    d(basis)


def d_cov(basis: Test[_L_co]) -> None:
    d(basis)


def d_con(basis: Test[_L_contra]) -> None:
    d(basis)


def e_inv(basis: Test[_LInv]) -> None:
    e(basis)


def e_cov(basis: Test[_L_co]) -> None:
    e(basis)


def e_con(basis: Test[_L_contra]) -> None:
    e(basis)
