"""
This type stub file was generated by pyright.
"""

from ._matrix import spmatrix
from ._base import _spbase, sparray
from ._index import IndexMixin

"""Dictionary Of Keys based matrix"""
__docformat__ = ...
__all__ = ['dok_array', 'dok_matrix', 'isspmatrix_dok']
class _dok_base(_spbase, IndexMixin, dict):
    """
    Dictionary Of Keys based sparse matrix.

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        dok_array(D)
            with a dense matrix, D

        dok_array(S)
            with a sparse matrix, S

        dok_array((M,N), [dtype])
            create the matrix with initial shape (M,N)
            dtype is optional, defaulting to dtype='d'

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Allows for efficient O(1) access of individual elements.
    Duplicates are not allowed.
    Can be efficiently converted to a coo_matrix once constructed.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import dok_array
    >>> S = dok_array((5, 5), dtype=np.float32)
    >>> for i in range(5):
    ...     for j in range(5):
    ...         S[i, j] = i + j    # Update element

    """
    _format = ...
    def __init__(self, arg1, shape=..., dtype=..., copy=...) -> None:
        ...
    
    def update(self, val):
        ...
    
    def count_nonzero(self): # -> int:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def get(self, key, default=...):
        """This overrides the dict.get method, providing type checking
        but otherwise equivalent functionality.
        """
        ...
    
    def __add__(self, other): # -> _NotImplementedType | dok_array:
        ...
    
    def __radd__(self, other): # -> _NotImplementedType | dok_array:
        ...
    
    def __neg__(self): # -> dok_array:
        ...
    
    def __imul__(self, other): # -> Self@_dok_base | _NotImplementedType:
        ...
    
    def __truediv__(self, other): # -> dok_array:
        ...
    
    def __itruediv__(self, other): # -> Self@_dok_base | _NotImplementedType:
        ...
    
    def __reduce__(self): # -> str | tuple[Any, ...]:
        ...
    
    def transpose(self, axes=..., copy=...): # -> dok_array:
        ...
    
    def conjtransp(self): # -> dok_array:
        """Return the conjugate transpose."""
        ...
    
    def copy(self): # -> dok_array:
        ...
    
    def tocoo(self, copy=...): # -> coo_array:
        ...
    
    def todok(self, copy=...): # -> dok_array | Self@_dok_base:
        ...
    
    def tocsc(self, copy=...): # -> csc_array:
        ...
    
    def resize(self, *shape): # -> None:
        ...
    


def isspmatrix_dok(x): # -> bool:
    """Is `x` of dok_array type?

    Parameters
    ----------
    x
        object to check for being a dok matrix

    Returns
    -------
    bool
        True if `x` is a dok matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dok_array, dok_matrix, coo_matrix, isspmatrix_dok
    >>> isspmatrix_dok(dok_matrix([[5]]))
    True
    >>> isspmatrix_dok(dok_array([[5]]))
    False
    >>> isspmatrix_dok(coo_matrix([[5]]))
    False
    """
    ...

class dok_array(_dok_base, sparray):
    ...


class dok_matrix(spmatrix, _dok_base):
    def set_shape(self, shape): # -> None:
        ...
    
    def get_shape(self): # -> tuple[int, int] | tuple[int | float, int | float]:
        """Get shape of a sparse array."""
        ...
    
    shape = ...


