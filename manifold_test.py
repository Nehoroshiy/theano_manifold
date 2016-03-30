import numpy as np

import time
import theano
import lasagne

from manifolds import FixedRankEmbeeded, ManifoldElement, TangentVector

import theano.tensor as T


class DotOp(theano.gof.Op):
    __props__ = ('manifold',)

    def __init__(self, manifold):
        super(DotOp, self).__init__()
        self.manifold = manifold

    def make_node(self, x, w):
        return theano.gof.graph.Apply(self, [x, w], [T.dmatrix()])

    def perform(self, node, inputs, output_storage):
        xin, w = inputs
        if xin.ndim > 2:
            xin = xin.reshape((xin.shape[0], -1))
        xout, = output_storage
        xout[0] = xin.dot(w.U).dot(w.S).dot(w.V)

    def grad(self, input, output_gradients):
        xin, w = input
        if xin.ndim > 2:
            xin = xin.reshape((xin.shape[0], -1))
        out_grad, = output_gradients

        xin_grad = out_grad[0].dot(w.V.T).dot(w.S.T).dot(w.U.T).reshape(xin.shape)
        w_grad = xin.reshape((xin.shape[0], -1)).T.dot(out_grad)
        print(type(w), type(w_grad), type(xin), type(out_grad))
        w_grad_projected = self.manifold.proj(w, w_grad)
        #w_grad_shared = theano.compile.SharedVariable(name="w_grad", type=ftan, value=w_grad_projected, strict=True)
        #w_grad_shared.ndim = 2

        return [xin_grad, w_grad_projected]

class DotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(), **kwargs):
        super(DotLayer, self).__init__(incoming, **kwargs)

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.manifold = FixedRankEmbeeded(num_inputs, num_units, min(num_inputs, num_units))

        self.num_units = num_units
        #U, S, V = T.nlinalg.svd(W.get_value(), full_matrices=False)
        #W_shared = theano.compile.SharedVariable(name="W", type=fman, value=self.manifold.rand(), strict=True)
        #W_shared.ndim = 2
        self.W = self.add_param(self.manifold.rand(), (num_inputs, num_units), name="W", regularizable=False)
        print(self.params.keys())
        print(self.get_params())
        print(self.get_params(**{"trainable": True}))

        #self.W = self.add_param(W, (num_inputs, num_units), name="W")

        self.op = DotOp(self.manifold)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return self.op(input, self.W)



if __name__ == "__main__":
    from mnist import load_dataset
    X_train,y_train,X_val,y_val,X_test,y_test = load_dataset()
    print(X_train.shape,y_train.shape)

    input_X = T.tensor4("X")
    input_shape = [None,1,28,28]
    target_y = T.vector("target Y integer", dtype='int32')

    #входной слой (вспомогательный)
    input_layer = lasagne.layers.InputLayer(shape = input_shape,input_var=input_X)

    #полносвязный слой, который принимает на вход input layer и имеет 100 нейронов.
    # нелинейная функция - сигмоида как в логистической регрессии
    # слоям тоже можно давать имена, но это необязательно
    dense_1 = DotLayer(input_layer,
                       num_units=10,
                       name = "output")
    biased_1 = lasagne.layers.BiasLayer(dense_1)
    nonlin_1 = lasagne.layers.NonlinearityLayer(biased_1, lasagne.nonlinearities.softmax)

    dense_1 = lasagne.layers.DenseLayer(input_layer,num_units=25,
                                       nonlinearity = lasagne.nonlinearities.sigmoid,
                                       name = "hidden_dense_layer")

    #ВЫХОДНОЙ полносвязный слой, который принимает на вход dense_1 и имеет 10 нейронов -по нейрону на цифру
    #нелинейность - softmax - чтобы вероятности всех цифр давали в сумме 1
    #dense_output = lasagne.layers.DenseLayer(dense_1,num_units = 10,
    #                                        nonlinearity = lasagne.nonlinearities.softmax,
    #                                       name='output')

    #предсказание нейронки (theano-преобразование)
    y_predicted = lasagne.layers.get_output(nonlin_1)

    #все веса нейронки (shared-переменные)
    all_weights = lasagne.layers.get_all_params(nonlin_1)
    print(all_weights)

    #функция ошибки - средняя кроссэнтропия
    loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()

    #сразу посчитать словарь обновлённых значений с шагом по градиенту, как раньше
    updates_sgd = lasagne.updates.sgd(loss, all_weights,learning_rate=0.01)

    #функция, которая обучает сеть на 1 шаг и возвращащет значение функции потерь и точности
    theano.config.exception_verbosity = 'high'

    theano.function([input_X,target_y],[loss,accuracy],updates=updates_sgd)