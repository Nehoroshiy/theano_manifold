

import time
import theano
import numpy as np
from numpy import linalg as la, random as rnd

from theano.tensor.shared_randomstreams import RandomStreams

srnd = RandomStreams(rnd.randint(0, 1000))

import theano.tensor as T

from manifolds import Oblique

if __name__ == "__main__":
    manifold = Oblique(10, 5)
    w = manifold.rand()
    ethalon = srnd.normal(size=(10, 5))
    egrad = srnd.normal(size=(10, 5))

    v_proj = manifold.egrad2rgrad(w, egrad)

    w_plus_v_proj = manifold.retr(w, v_proj)

    w_example = theano.function([], w.norm(L=2, axis=0))
    update_example = theano.function([], w_plus_v_proj.norm(L=2, axis=0))
    print(w_example())
    print(update_example())

    loss = T.sum((ethalon - w)**2)
    floss = theano.function([], loss)
    w_egrad, = theano.grad(loss, [w])

    w_rgrad = manifold.egrad2rgrad(w, w_egrad)

    w_plus_rgrad = manifold.retr(w, - 0.001*w_rgrad)

    updated_weight_on_loss = theano.function([], w_plus_rgrad.norm(L=2, axis=0))

    print("loss before: {}".format(floss()))
    w = manifold.retr(w, -0.001 * w_rgrad)
    print("loss after: {}".format(floss()))

    print(updated_weight_on_loss())
