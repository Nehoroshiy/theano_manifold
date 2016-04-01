import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))

import warnings
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from manifolds.manifold import Manifold

import copy

import theano
from theano import tensor
from theano.gof import Container, Variable


class ManifoldElementShared(theano.compile.SharedVariable):
    def __init__(self, value, name=None, type=theano.tensor.dmatrix):
        if isinstance(value, tuple):
            if len(value) == 2:
                if isinstance(value[0], theano.compile.SharedVariable) and isinstance(value[1], tuple):
                    value, shape = value
                    self._m, self._n = shape
                    self._r = value.get_value(borrow=True).shape[1]
                    super(ManifoldElementShared, self).__init__(name=name,
                                                        type=type,
                                                        value=value.get_value(), strict=True)
            elif len(value) == 3 and all([isinstance(item, np.ndarray) for item in value]):
                U, S, V = value
                self._m = U.shape[0]
                self._n = V.shape[1]
                self._r = S.shape[0]
                super(ManifoldElementShared, self).__init__(name=name,
                                                        type=type,
                                                        value=np.vstack([U, S, V.T]), strict=True)
            else:
                raise TypeError("value must be a tuple(SharedVariable, shape) or a tuple of 3 ndarrays Up, M, Vp")

        else:
            raise TypeError("value must be tuple(SharedVariable, shape) or tuple of 3 ndarrays")
        self.update_factor_view()

    def update_factor_view(self):
        self.U = theano.shared(self.get_value(borrow=True)[:self._m, :],
                               borrow=True,
                               name="U")
        self.S = theano.shared(self.get_value(borrow=True)[self._m: self._m + self._r, :],
                               borrow=True,
                               name="S")
        self.V = theano.shared(self.get_value(borrow=True)[self._m + self._r :, :].T,
                               borrow=True,
                               name="V")

    def set_value(self, new_value, borrow=False):
        """
        Set the non-symbolic value associated with this SharedVariable.
        Parameters
        ----------
        borrow : bool
            True to use the new_value directly, potentially creating problems
            related to aliased memory.
        Changes to this value will be visible to all functions using
        this SharedVariable.
        """
        if borrow:
            self.container.value = new_value
        else:
            self.container.value = copy.deepcopy(new_value)
        self.update_factor_view()

    @property
    def r(self):
        return self._r

    @property
    def shape(self):
        return (self._m, self._n)

    @property
    def ndim(self):
        return len(self.shape)

    def clone(self):
        """
        Return a new Variable like self.
        Returns
        -------
        Variable instance
            A new Variable instance (or subclass instance) with no owner or
            index.
        Notes
        -----
        Tags are copied to the returned instance.
        Name is copied to the returned instance.
        """
        # return copy(self)
        cp = self.__class__((self.U.get_value(), self.S.get_value(), self.V.get_value()), name=self.name, type=self.type)
        #cp = self.__class__(self.type, None, None, self.name)
        #cp.tag = copy(self.tag)
        return cp

    @classmethod
    def from_vars(cls, values, shape, r, name=None, type=theano.tensor.dmatrix):
        u, s, v = values
        value = theano.shared(np.zeros((shape[0] + shape[1] + r, r)))
        new_value = value
        new_value = theano.tensor.set_subtensor(new_value[:shape[0]], u)
        new_value = theano.tensor.set_subtensor(new_value[shape[0]: shape[0] + r], s)
        new_value = theano.tensor.set_subtensor(new_value[shape[0] + r :], v)
        updates = [(value, new_value)]
        func = theano.function(inputs=[], outputs=[], updates=updates)
        func()
        return cls((value, shape), name=name, type=type)


class TangentVectorShared(theano.compile.SharedVariable):
    def __init__(self, value, name=None, type=theano.tensor.dmatrix):
        if isinstance(value, tuple):
            if len(value) == 2:
                if isinstance(value[0], theano.compile.SharedVariable) and isinstance(value[1], tuple):
                    value, shape = value
                    self._m, self._n = shape
                    self._r = value.get_value(borrow=True).shape[1]
                    super(TangentVectorShared, self).__init__(name=name,
                                                        type=type,
                                                        value=value.get_value(), strict=True)
            elif len(value) == 3 and all([isinstance(item, np.ndarray) for item in value]):
                Up, M, Vp = value
                self._m = Up.shape[0]
                self._n = Vp.shape[1]
                self._r = M.shape[0]
                super(TangentVectorShared, self).__init__(name=name,
                                                        type=type,
                                                        value=np.vstack([Up, M, Vp.T]), strict=True)
            else:
                raise TypeError("value must be a tuple(SharedVariable, shape) or a tuple of 3 ndarrays Up, M, Vp")

        else:
            raise TypeError("value must be tuple(SharedVariable, shape) or tuple of 3 ndarrays")
        self.update_factor_view()

    def update_factor_view(self):
        self.Up = theano.shared(self.get_value(borrow=True)[:self._m, :],
                               borrow=True,
                               name="Up")
        self.M = theano.shared(self.get_value(borrow=True)[self._m: self._m + self._r, :],
                               borrow=True,
                               name="M")
        self.Vp = theano.shared(self.get_value(borrow=True)[self._m + self._r :, :].T,
                               borrow=True,
                               name="Vp")

    def set_value(self, new_value, borrow=False):
        """
        Set the non-symbolic value associated with this SharedVariable.
        Parameters
        ----------
        borrow : bool
            True to use the new_value directly, potentially creating problems
            related to aliased memory.
        Changes to this value will be visible to all functions using
        this SharedVariable.
        """
        if borrow:
            self.container.value = new_value
        else:
            self.container.value = copy.deepcopy(new_value)
        self.update_factor_view()

    @property
    def r(self):
        return self._r

    @property
    def shape(self):
        return (self._m, self._n)

    @property
    def ndim(self):
        return len(self.shape)

    def clone(self):
        """
        Return a new Variable like self.
        Returns
        -------
        Variable instance
            A new Variable instance (or subclass instance) with no owner or
            index.
        Notes
        -----
        Tags are copied to the returned instance.
        Name is copied to the returned instance.
        """
        # return copy(self)
        cp = self.__class__((self.Up.get_value(), self.M.get_value(), self.Vp.get_value()), name=self.name, type=self.type)
        #cp = self.__class__(self.type, None, None, self.name)
        #cp.tag = copy(self.tag)
        return cp

    @classmethod
    def from_vars(cls, values, shape, r, name=None, type=theano.tensor.dmatrix):
        u, s, v = values
        value = theano.shared(np.zeros((shape[0] + shape[1] + r, r)))
        new_value = value
        new_value = theano.tensor.set_subtensor(new_value[:shape[0]], u)
        new_value = theano.tensor.set_subtensor(new_value[shape[0]: shape[0] + r], s)
        new_value = theano.tensor.set_subtensor(new_value[shape[0] + r :], v)
        updates = [(value, new_value)]
        func = theano.function(inputs=[], outputs=[], updates=updates)
        func()
        return cls((value, shape), name=name, type=type)


class FixedRankEmbeeded(Manifold):
    """
    Manifold of m-by-n real matrices of fixed rank k. This follows the
    embedded geometry described in Bart Vandereycken's 2013 paper:
    "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    A point X on the manifold is represented as a structure with three
    fields: U, S and V. The matrices U (mxk) and V (kxn) are orthonormal,
    while the matrix S (kxk) is any /diagonal/, full rank matrix.
    Following the SVD formalism, X = U*S*V. Note that the diagonal entries
    of S are not constrained to be nonnegative.

    Tangent vectors are represented as a structure with three fields: Up, M
    and Vp. The matrices Up (mxn) and Vp (kxn) obey Up*U = 0 and Vp*V = 0.
    The matrix M (kxk) is arbitrary. Such a structure corresponds to the
    following tangent vector in the ambient space of mxn matrices:
      Z = U*M*V + Up*V + U*Vp
    where (U, S, V) is the current point and (Up, M, Vp) is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as mxn matrices. If
    these are low-rank, they may also be represented as structures with
    U, S, V fields, such that Z = U*S*V. Their are no resitrictions on what
    U, S and V are, as long as their product as indicated yields a real, mxn
    matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space R^(mxn) equipped with the usual trace (Frobenius) inner product.


    Please cite the Manopt paper as well as the research paper:
        @Article{vandereycken2013lowrank,
          Title   = {Low-rank matrix completion by {Riemannian} optimization},
          Author  = {Vandereycken, B.},
          Journal = {SIAM Journal on Optimization},
          Year    = {2013},
          Number  = {2},
          Pages   = {1214--1236},
          Volume  = {23},
          Doi     = {10.1137/110845768}
        }
    """
    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k
        #self.stiefelm = Stiefel(self._m, self._k)
        #self.stiefeln = Stiefel(self._n, self._k)
        self._name = ('Manifold of {:d}x{:d} matrices of rank {:d}'.format(m, n, k))

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return (self._m + self._n - self._k) * self._k

    @property
    def typicaldist(self):
        return self.dim

    def inner(self, X, G, H):
        return G.M.ravel().dot(H.M.ravel()) + \
               G.Up.ravel().dot(H.Up.ravel()) + \
               G.Vp.ravel().dot(H.Vp.ravel())

    def norm(self, X, G):
        return tensor.sqrt(self.inner(X, G, G))

    def dist(self, X, Y):
        raise NotImplementedError

    def tangent(self, X, Z):
        # ??? how can we perform inplace operations?
        # TODO make inplace update for Z.Up and Z.Vp
        # like this:
        # s.set_value(
        #             some_inplace_fn(s.get_value(borrow=True)),
        #             borrow=True)

        #Z.Up = Z.Up - X.U.dot(X.U.T.dot(Z.Up))
        #Z.Vp = Z.Vp - (Z.Vp.dot(X.V.T)).dot(X.V)
        raise NotImplementedError("method is not imlemented")

    def apply_ambient(self, Z, W, type="mat"):
        if isinstance(Z, ManifoldElementShared):
            return Z.U.dot(Z.S.dot(Z.V.dot(W)))
        if isinstance(Z, TangentVectorShared):
            return Z.Up.dot(Z.M.dot(Z.Vp.dot(W)))
        else:
            return Z.dot(W)

    def apply_ambient_transpose(self, Z, W):
        if isinstance(Z, ManifoldElementShared):
            return Z.V.T.dot(Z.S.T.dot(Z.U.T.dot(W)))
        if isinstance(Z, TangentVectorShared):
            return Z.Vp.T.dot(Z.M.T.dot(Z.Up.T.dot(W)))
        else:
            return Z.T.dot(W)

    def proj(self, X, Z):
        ZV = self.apply_ambient(Z, X.V.T)
        UtZV = X.U.T.dot(ZV)
        ZtU = self.apply_ambient_transpose(Z, X.U).T

        Zproj = TangentVectorShared.from_vars((ZV - X.U.dot(UtZV), UtZV, ZtU - (UtZV.dot(X.V))),
                                              shape=(self._m, self._n), r=self._k)
        return Zproj

    def egrad2rgrad(self, X, Z):
        return self.proj(X, Z)

    def ehess2rhess(self, X, egrad, ehess, H):
        # TODO same problem as tangent
        """
        # Euclidean part
        rhess = self.proj(X, ehess)
        Sinv = tensor.diag(1.0 / tensor.diag(X.S))

        # Curvature part
        T = self.apply_ambient(egrad, H.Vp.T).dot(Sinv)
        rhess.Up += (T - X.U.dot(X.U.T.dot(T)))
        T = self.apply_ambient_transpose(egrad, H.Up).dot(Sinv)
        rhess.Vp += (T - X.V.T.dot(X.V.dot(T))).T
        return rhess
        """
        raise NotImplementedError("method is not imlemented")

    def tangent2ambient(self, X, Z):
        U = tensor.stack((X.U.dot(Z.M) + Z.Up, X.U), 0).reshape((-1, X.U.shape[1]))
        #U = np.hstack((X.U.dot(Z.M) + Z.Up, X.U))
        S = tensor.eye(2*self._k)
        V = tensor.stack((X.V, Z.Vp), 1).reshape((X.V.shape[0], -1))
        #V = np.vstack((X.V, Z.Vp))
        return ManifoldElementShared.from_vars((U, S, V), shape=(self._m, self._n), r=self._k)

    def retr(self, X, Z, t=None):
        if t is None:
            t = 1.0
        Qu, Ru = tensor.nlinalg.QRFull(Z.Up)

        # we need rq decomposition here
        Qv, Rv = tensor.nlinalg.QRFull(Z.Vp[::-1].T)
        Rv = Rv.T[::-1]
        Rv[:, :] = Rv[:, ::-1]
        Qv = Qv.T[::-1]

        # now we have rq decomposition (Rv @ Qv = Z.Vp)
        #Rv, Qv = rq(Z.Vp, mode='economic')


        zero_block = tensor.zeros((Ru.shape[0], Rv.shape[1]))
        block_mat = tensor.stack(
            (
                tensor.stack((X.S + t * Z.M, t * Rv), 1).reshape((Rv.shape[0], -1)),
                tensor.stack((t * Ru, zero_block), 1).reshape((Ru.shape[0], -1))
            )
        ).reshape((-1, Ru.shape[1] + Rv.shape[1]))

        Ut, St, Vt = tensor.nlinalg.svd(block_mat, full_matrices=False)

        U = tensor.stack((X.U, Qu), 1).reshape((Qu.shape[0], -1)).dot(Ut[:, :self._k])
        V = Vt[:self._k, :].dot(tensor.stack((X.V, Qv), 0).reshape((-1, Qv.shape[1])))
        # add some machinery eps to get a slightly perturbed element of a manifold
        # even if we have some zeros in S
        S = tensor.diag(St[:self._k]) + tensor.diag(np.spacing(1) * tensor.ones(self._k))
        return ManifoldElementShared.from_vars((U, S, V), shape=(self._m, self._n), r=self._k)

    def exp(self, X, U, t=None):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U, t)

    def np_rand(self, shape):
        X = np.random.randn(*shape)
        q, r = np.linalg.qr(X)
        return q

    def rand(self, name=None):
        s = np.sort(np.random.random(self._k))[::-1]
        S = np.diag(s / la.norm(s) + np.spacing(1) * np.ones(self._k))
        return ManifoldElementShared((rnd.randn(self._m, self._k), S, rnd.randn(self._k, self._n)), name=name)

    def randvec(self, X):
        H = self.rand()
        P = self.proj(X, H)
        return self._normalize(P)

    def zerovec(self, X):
        return TangentVectorShared(np.zeros((self._m, self._k)),
                                np.zeros((self._k, self._k)),
                                np.zeros((self._k, self._n)))

    def vec(self, X, Z):
        Zamb = self.tangent2ambient(X, Z)
        Zamb_mat = Zamb.U.dot(Zamb.S).dot(Zamb.V)
        Zvec = Zamb_mat.T.ravel()
        return Zvec

    def _normalize(self, P):
        Up = P.Up
        M = P.M / tensor.nlinalg.norm(P.M)
        Vp = P.Vp
        return TangentVectorShared.from_vars((Up, M, Vp), shape=(self._m, self._n), r=self._k)

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d))

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            Up = a1 * u1.Up
            Vp = a1 * u1.Vp
            M = a1 * u1.M
            return TangentVectorShared.from_vars((Up, M, Vp), shape=(self._m, self._n), r=self._k)
        elif None not in [a1, u1, a2, u2]:
            Up = a1 * u1.Up + a2 * u2.Up
            Vp = a1 * u1.Vp + a2 * u2.Vp
            M = a1 * u1.M + a2 * u2.M
            return TangentVectorShared.from_vars((Up, M, Vp), shape=(self._m, self._n), r=self._k)
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')


