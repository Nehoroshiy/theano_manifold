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
    Rotations manifold: deals with matrices X of size n-by-n such that X^T X = I.
    """

    def __init__(self, n):
        self._n = n

    def __str__(self):
        return "Rotations manifold SO({:d})".format(self._n)

    @property
    def dim(self):
        return self._n * self._n