from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple

import torch
# necessary for custom torch.svd_lowrank realization
from torch import Tensor
from torch import _linalg_utils as _utils

# get_approximate_basis() and _svd_lowrank() functions are taken from torch library
# with a minor correction allowing not to store large matrices when input matrices are sparse
def get_approximate_basis(A,        # type: Tensor
                          q,        # type: int
                          niter=2,  # type: Optional[int]
                          M=None    # type: Optional[Tensor]
                          ):
    # type: (...) -> Tensor
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.
    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.
    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.
    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator
    Arguments::
        A (Tensor): the input tensor of size :math:`(*, m, n)`
        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.
        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.
        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.
    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    m, n = A.shape[-2:]
    dtype = _utils.get_floating_dtype(A)
    matmul = _utils.matmul

    R = torch.randn(n, q, dtype=dtype, device=A.device)

    A_H = _utils.transjugate(A)
    if M is None:
        (Q, _) = torch.linalg.qr(matmul(A, R))#.qr()
        for i in range(niter):
            # (Q, _) = matmul(A_H, Q).qr()
            # (Q, _) = matmul(A, Q).qr()
            (Q, _) = torch.linalg.qr(matmul(A_H, Q))
            (Q, _) = torch.linalg.qr(matmul(A, Q))
    else:
        M_H = _utils.transjugate(M)
        (Q, _) = torch.linalg.qr(matmul(A, R) - matmul(M, R))#.qr()
        for i in range(niter):
            (Q, _) = torch.linalg.qr(matmul(A_H, Q) - matmul(M_H, Q))#.qr()
            (Q, _) = torch.linalg.qr(matmul(A, Q) - matmul(M, Q))#.qr()

    return Q


def _svd_lowrank(A, q=6, niter=2, M=None):
    # type: (Tensor, Optional[int], Optional[int], Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    q = 6 if q is None else q
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is None:
        M_t = None
    else:
        M_t = _utils.transpose(M)
    A_t = _utils.transpose(A)

    # Algorithm 5.1 in Halko et al 2009, slightly modified to reduce
    # the number conjugate and transpose operations
    if m < n or n > q:
        # computing the SVD approximation of a transpose in order to
        # keep B shape minimal
        Q = get_approximate_basis(A_t, q, niter=niter, M=M_t)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B_t = matmul(A, Q_c)
        else:
            B_t = matmul(A, Q_c) - matmul(M, Q_c)
        assert B_t.shape[-2] == m, (B_t.shape, m)
        assert B_t.shape[-1] == q, (B_t.shape, q)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, V = torch.svd(B_t)
        V = Q.matmul(V)
    else:
        Q = get_approximate_basis(A, q, niter=niter, M=M)
        Q_c = _utils.conjugate(Q)
        if M is None:
            B = matmul(A_t, Q_c)
        else:
            B = matmul(A_t, Q_c) - matmul(M_t, Q_c)
        B_t = _utils.transpose(B)
        assert B_t.shape[-2] == q, (B_t.shape, q)
        assert B_t.shape[-1] == n, (B_t.shape, n)
        assert B_t.shape[-1] <= B_t.shape[-2], B_t.shape
        U, S, V = torch.svd(B_t)
        U = Q.matmul(U)

    return U, S, V
