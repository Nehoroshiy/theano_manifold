import warnings
import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from scipy.sparse import coo_matrix
from scipy.linalg import solve_lyapunov as lyap, rq

from manifolds.manifold import Manifold

import copy

import theano
from theano import tensor
from theano.gof import Container, Variable



class SharedManifoldVariable(theano.compile.SharedVariable):
    """
    Variable that is (defaults to being) shared between functions that
    it appears in.
    Parameters
    ----------
    name : str
        The name for this variable (see `Variable`).
    type : str
        The type for this variable (see `Variable`).
    value
        A value to associate with this variable (a new container will be
        created).
    strict
        True : assignments to .value will not be cast or copied, so they must
        have the correct type.
    allow_downcast
        Only applies if `strict` is False.
        True : allow assigned value to lose precision when cast during
        assignment.
        False : never allow precision loss.
        None : only allow downcasting of a Python float to a scalar floatX.
    container
        The container to use for this variable. Illegal to pass this as well as
        a value.
    Notes
    -----
    For more user-friendly constructor, see `shared`.
    """

    # Container object
    container = None
    """
    A container to use for this SharedVariable when it is an implicit
    function parameter.
    :type: `Container`
    """

    # default_update
    # If this member is present, its value will be used as the "update" for
    # this Variable, unless another update value has been passed to "function",
    # or the "no_default_updates" list passed to "function" contains it.

    def __init__(self, name, type, value, strict,
                 allow_downcast=None, container=None):
        super(SharedManifoldVariable, self).__init__(type=type, name=name)

        if container is not None:
            self.container = container
            if (value is not None) or (strict is not None):
                raise TypeError('value and strict are ignored if you pass '
                                'a container here')
        else:
            if container is not None:
                raise TypeError('Error to specify both value and container')
            self.container = Container(
                self,
                storage=[type.filter(value, strict=strict,
                                     allow_downcast=allow_downcast)],
                readonly=False,
                strict=strict,
                allow_downcast=allow_downcast)

    def get_value(self, borrow=False, return_internal_type=False):
        """
        Get the non-symbolic value associated with this SharedVariable.
        Parameters
        ----------
        borrow : bool
            True to permit returning of an object aliased to internal memory.
        return_internal_type : bool
            True to permit the returning of an arbitrary type object used
            internally to store the shared variable.
        Only with borrow=False and return_internal_type=True does this function
        guarantee that you actually get the internal object.
        But in that case, you may get different return types when using
        different compute devices.
        """
        if borrow:
            return self.container.value
        else:
            return copy.deepcopy(self.container.value)

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

    def zero(self, borrow=False):
        """
        Set the values of a shared variable to 0.
        Parameters
        ----------
        borrow : bbol
            True to modify the value of a shared variable directly by using
            its previous value. Potentially this can cause problems
            regarding to the aliased memory.
        Changes done with this function will be visible to all functions using
        this SharedVariable.
        """
        if not isinstance(self.container.value, ManifoldElement):
            raise TypeError("underlying value must be ManifoldElement!")
        if borrow:
            self.container.value.make_zero()
        else:
            self.container.value = FixedRankEmbeeded(*self.container.value.shape, self.container.value.r).rand().make_zero()

    def clone(self):
        cp = self.__class__(
            name=self.name,
            type=self.type,
            value=None,
            strict=None,
            container=self.container)
        cp.tag = copy.copy(self.tag)
        return cp

    def __getitem__(self, *args):
        # __getitem__ is not available for generic SharedVariable objects.
        # We raise a TypeError like Python would do if __getitem__ was not
        # implemented at all, but with a more explicit error message to help
        # Theano users figure out the root of the problem more easily.
        value = self.get_value(borrow=True)
        if isinstance(value, ManifoldElement):
            # Array probably had an unknown dtype.
            msg = ("a ManifoldElement array with dtype: '%s'. This data type is not "
                   "currently recognized by Theano tensors: please cast "
                   "your data into a supported numeric type if you need "
                   "Theano tensor functionalities." % value.dtype)
        else:
            msg = ('an object of type: %s. Did you forget to cast it into '
                   'a Numpy array before calling theano.shared()?' %
                   type(value))

        raise TypeError(
            "The generic 'SharedVariable' object is not subscriptable. "
            "This shared variable contains %s" % msg)

    def _value_get(self):
        raise Exception("sharedvar.value does not exist anymore. Use "
                        "sharedvar.get_value() or sharedvar.set_value()"
                        " instead.")

    def _value_set(self, new_value):
        raise Exception("sharedvar.value does not exist anymore. Use "
                        "sharedvar.get_value() or sharedvar.set_value()"
                        " instead.")

    # We keep this just to raise an error
    value = property(_value_get, _value_set)


class FixedRankManifoldType(theano.tensor.type.TensorType):#theano.gof.Type):
    def __init__(self):
        #super(FixedRankManifoldType, self).__init__(dtype=np.float, broadcastable=(False, False))
        self.dtype='floatX'#np.array([1.0]).dtype
        self.broadcastable=(False, False)

    def filter(self, x, strict=True, allow_downcast=None):
        if strict:
            if isinstance(x, tensor.Variable):
                u, s, v = tensor.nlinalg.svd(x, full_matrices=False)
                return ManifoldElement(u, tensor.diag(s), v)
            if isinstance(x, np.ndarray):
                u, s, v = np.linalg.svd(x, full_matrices=False)
                return ManifoldElement(theano.shared(u), theano.shared(np.diag(s)), theano.shared(v))
            if isinstance(x, ManifoldElement):
                return ManifoldElement(x.U, x.S, x.V)
            else:
                raise TypeError('Expected an symbolc tensor variable, ndarray or ManifoldElement!')
        elif allow_downcast:
            raise TypeError('downcast is not allowed!')
            u, s, v = np.linalg.svd(np.array(x, dtype=float), full_matrices=False)
            return ManifoldElement(u, np.diag(s), v)
        else:   # Covers both the False and None cases.
             raise TypeError('The double type cannot accurately represent '
                             'value %s (of type %s): you must explicitly '
                             'allow downcasting if you want to do this.'
                             % (x, type(x)))

    def __str__(self):
        return "FixedRankManifoldType"

fman = FixedRankManifoldType()


class ManifoldElement(Variable):
    def __init__(self, U, S, V):
        super(ManifoldElement, self).__init__(fman)
        self.U = U.copy()
        self.S = S.copy()
        self.V = V.copy()
        self.r = S.shape[0]
        self.shape=(U.shape[0], V.shape[1])
        self.ndim = len(self.shape)

    def __add__(self, other):
        if isinstance(other, TangentVector):
            return FixedRankEmbeeded(self.U.shape[0], self.V.shape[1], self.S.shape[0]).retr(self, other)

    def __sub__(self, other):
        if isinstance(other, TangentVector):
            return FixedRankEmbeeded(self.U.shape[0], self.V.shape[1], self.S.shape[0]).retr(self, other, -1.0)

    def dot(self, other):
        if isinstance(other, ManifoldElement):
            mid = self.S.dot(self.V.dot(other.U)).dot(other.S)
            U, S, V = tensor.nlinalg.svd(mid, full_matrices=False)
            return ManifoldElement(self.U.dot(U), tensor.diag(self.S), V.dot(self.V))
        else:
            raise ValueError('dot must be performed on ManifoldElements.')

    """
    def __getitem__(self, item):
        if hasattr(item, '__len__') and len(item) == 2 and len(item[0]) == len(item[1]):
            rows = self.U[item[0], :].dot(self.S)
            cols = self.V[:, item[1]]
            data = (rows * cols.T).sum(1)
            #assert(data.size == len(item[0]))
            shape = (self.U.shape[0], self.V.shape[1])
            return coo_matrix((data, tuple(item)), shape=shape).tocsr()
        else:
            raise ValueError('__getitem__ now supports only indices set')
    """

    def full(self):
        return self.U.dot(self.S).dot(self.V)

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
        cp = self.__class__(self.U, self.S, self.V)
        #cp = self.__class__(self.type, None, None, self.name)
        #cp.tag = copy(self.tag)
        return cp

    def make_zero(self):
        self.U = tensor.zeros_like(self.U)
        self.S = tensor.zeros_like(self.S)
        self.V = tensor.zeros_like(self.V)

    @property
    def T(self):
        return ManifoldElement(self.V.T, self.S.T, self.U.T)


class FixedRankTangentType(theano.tensor.type.TensorType):#theano.gof.Type):
    def __init__(self):
        #super(FixedRankManifoldType, self).__init__(dtype=np.float, broadcastable=(False, False))
        self.dtype='floatX'#np.array([1.0]).dtype
        self.broadcastable=(False, False)

    def filter(self, x, strict=True, allow_downcast=None):
        if strict:
            if isinstance(x, TangentVector):
                return TangentVector(x.Up, x.M, x.Vp)
            else:
                raise TypeError('Expected a TangentVector!')
        elif allow_downcast:
            raise TypeError('cannot downcast into TangentVector')
        else:   # Covers both the False and None cases.
             raise TypeError('The double type cannot accurately represent '
                             'value %s (of type %s): you must explicitly '
                             'allow downcasting if you want to do this.'
                             % (x, type(x)))

    def __str__(self):
        return "FixedRankManifoldType"

ftan = FixedRankTangentType()

class TangentVector(Variable):
    def __init__(self, Up, M, Vp):
        super(TangentVector, self).__init__(type(self))
        self.Up = Up.copy()
        self.M = M.copy()
        self.Vp = Vp.copy()
        self.shape=(Up.shape[0], Vp.shape[1])
        self.ndim = len(self.shape)

    def __neg__(self):
        return TangentVector(-self.Up, -self.M, -self.Vp)

    def __add__(self, other):
        if isinstance(other, TangentVector):
            return TangentVector(self.Up + other.Up, self.M + other.M, self.Vp + other.Vp)

    def __sub__(self, other):
        if isinstance(other, TangentVector):
            return TangentVector(self.Up - other.Up, self.M - other.M, self.Vp - other.Vp)

    def __mul__(self, other):
        if tensor.iscalar(other):
            return TangentVector(self.Up * other, self.M * other, self.Vp * other)
        else:
            raise ValueError('TangentVector supports only multiplying by scalar')

    def __rmul__(self, other):
        return self.__mul__(other)

    def make_zero(self):
        self.Up = tensor.zeros_like(self.Up)
        self.M = tensor.zeros_like(self.M)
        self.Vp = tensor.zeros_like(self.Vp)

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
        cp = self.__class__(self.Up, self.M, self.Vp)
        #cp = self.__class__(self.type, None, None, self.name)
        #cp.tag = copy(self.tag)
        return cp




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
        Z.Up = Z.Up - X.U.dot(X.U.T.dot(Z.Up))
        Z.Vp = Z.Vp - (Z.Vp.dot(X.V.T)).dot(X.V)

    def apply_ambient(self, Z, W):
        if isinstance(Z, ManifoldElement):
            return Z.U.dot(Z.S.dot(Z.V.dot(W)))
        if isinstance(Z, TangentVector):
            return Z.Up.dot(Z.M.dot(Z.Vp.dot(W)))
        else:
            return Z.dot(W)

    def apply_ambient_transpose(self, Z, W):
        if isinstance(Z, ManifoldElement):
            return Z.V.T.dot(Z.S.T.dot(Z.U.T.dot(W)))
        if isinstance(Z, TangentVector):
            return Z.Vp.T.dot(Z.M.T.dot(Z.Up.T.dot(W)))
        else:
            return Z.T.dot(W)

    def proj(self, X, Z):
        ZV = self.apply_ambient(Z, X.V.T)
        UtZV = X.U.T.dot(ZV)
        ZtU = self.apply_ambient_transpose(Z, X.U).T

        Zproj = TangentVector(ZV - X.U.dot(UtZV), UtZV, ZtU - (UtZV.dot(X.V)))
        return Zproj

    def egrad2rgrad(self, X, Z):
        return self.proj(X, Z)

    def ehess2rhess(self, X, egrad, ehess, H):
        # Euclidean part
        rhess = self.proj(X, ehess)
        Sinv = tensor.diag(1.0 / tensor.diag(X.S))

        # Curvature part
        T = self.apply_ambient(egrad, H.Vp.T).dot(Sinv)
        rhess.Up += (T - X.U.dot(X.U.T.dot(T)))
        T = self.apply_ambient_transpose(egrad, H.Up).dot(Sinv)
        rhess.Vp += (T - X.V.T.dot(X.V.dot(T))).T
        return rhess

    def tangent2ambient(self, X, Z):
        U = tensor.stack((X.U.dot(Z.M) + Z.Up, X.U), 0).reshape((-1, X.U.shape[1]))
        #U = np.hstack((X.U.dot(Z.M) + Z.Up, X.U))
        S = tensor.eye(2*self._k)
        V = tensor.stack((X.V, Z.Vp), 1).reshape((X.V.shape[0], -1))
        #V = np.vstack((X.V, Z.Vp))
        return ManifoldElement(U, S, V)

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
        return ManifoldElement(U, S, V)

    def exp(self, X, U, t=None):
        warnings.warn("Exponential map for fixed-rank matrix"
                      "manifold not implemented yet. Used retraction instead.",
                      RuntimeWarning)
        return self.retr(X, U, t)

    def np_rand(self, shape):
        X = np.random.randn(*shape)
        q, r = np.linalg.qr(X)
        return q

    def rand(self):
        U = theano.shared(self.np_rand((self._m, self._k)))
        V = theano.shared(self.np_rand((self._n, self._k)).T)
        s = np.sort(np.random.random(self._k))[::-1]
        S = np.diag(s / la.norm(s) + np.spacing(1) * np.ones(self._k))
        return ManifoldElement(U, theano.shared(S), V)

    def randvec(self, X):
        H = self.rand()
        P = self.proj(X, H)
        return self._normalize(P)

    def zerovec(self, X):
        return TangentVector(tensor.zeros((self._m, self._k)),
                                tensor.zeros((self._k, self._k)),
                                tensor.zeros((self._k, self._n)))

    def vec(self, X, Z):
        Zamb = self.tangent2ambient(X, Z)
        Zamb_mat = Zamb.U.dot(Zamb.S).dot(Zamb.V)
        Zvec = Zamb_mat.T.ravel()
        return Zvec

    def _normalize(self, P):
        Up = P.Up
        M = P.M / tensor.nlinalg.norm(P.M)
        Vp = P.Vp
        return TangentVector(Up, M, Vp)

    def log(self, X, Y):
        raise NotImplementedError

    def transp(self, x1, x2, d):
        return self.proj(x2, self.tangent2ambient(x1, d))

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        if u2 is None and a2 is None:
            Up = a1 * u1.Up
            Vp = a1 * u1.Vp
            M = a1 * u1.M
            return TangentVector(Up, M, Vp)
        elif None not in [a1, u1, a2, u2]:
            Up = a1 * u1.Up + a2 * u2.Up
            Vp = a1 * u1.Vp + a2 * u2.Vp
            M = a1 * u1.M + a2 * u2.M
            return TangentVector(Up, M, Vp)
        else:
            raise ValueError('FixedRankEmbeeded.lincomb takes 3 or 5 arguments')


