import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))

from manifolds.manifold import Manifold


class Rotations(Manifold):
    """
    Manifold of matrices from SO(n).
    Ootations manifold: deals with matrices of size m-by-n such that each column
    has unit 2-norm, i.e., is a point on the unit sphere in R^m. The metric
    is such that the oblique manifold is a Riemannian submanifold of the
    space of m-by-n matrices with the usual trace inner product, i.e., the
    usual metric.
    """

    def __init__(self, m, n):
        self._m = m
        self._n = n

    def __str__(self):
        return "Oblique manifold OB({:d}, {:d})".format(self._m, self._n)

