import numpy as np

import time
import theano
import lasagne

from manifolds import FixedRankEmbeeded
from collections import OrderedDict


import theano.tensor as T


class DotStep(theano.gof.Op):
    __props__ = ('manifold',)

    def __init__(self, manifold):
        super(DotStep, self).__init__()
        self.manifold = manifold

    def make_node(self, x, u, s, v):
        return theano.gof.graph.Apply(self, [x, u, s, v], [T.dmatrix()])

    def perform(self, node, inputs, output_storage):
        xin, u, s, v = inputs
        #print(np.linalg.norm(u), np.linalg.norm(s), np.linalg.norm(v))
        #print("w norm: {}".format(np.linalg.norm(w, axis=0)))
        #print(type(xin), xin.shape)

        if xin.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            xin = np.reshape(xin, (xin.shape[0], -1))
            #xin = xin.flatten(2)
        #print(xin)
        #print('_--_')

        activation = xin.dot(u).dot(s).dot(v)
        #print("activation norm:{}".format(np.linalg.norm(activation, axis=0)))
        xout, = output_storage
        xout[0] = activation

    def grad(self, input, output_gradients):
        xin, u, s, v = input
        #print("x: {}, w: {}".format(type(xin), type(w)))
        if xin.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            xin = xin.flatten(2)
        out_grad, = output_gradients

        # space issue --- maybe factorize xin and out_grad before dot
        w_egrad = xin.T.dot(out_grad)
        w_rgrad = self.manifold.egrad2rgrad((u, s, v), w_egrad)

        xin_grad = out_grad.dot(v.T).dot(s.T).dot(u.T).reshape(xin.shape)

        return [xin_grad, *w_rgrad]


class DotLayer2(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(), **kwargs):
        super(DotLayer2, self).__init__(incoming, **kwargs)

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.r = 3
        self.manifold = FixedRankEmbeeded(num_inputs, num_units, self.r)
        self.num_units = num_units
        U, S, V = self.manifold.rand_np()
        # give proper names
        self.U = self.add_param(U, (num_inputs, self.r), name="U", regularizable=False)
        self.S = self.add_param(S, (self.r, self.r), name="S", regularizable=True)
        self.V = self.add_param(V, (self.r, num_units), name="V", regularizable=False)
        #self.W = self.add_param(self.manifold.rand(name="USV"), (num_inputs, num_units), name="W")
        #self.W = self.manifold._normalize_columns(self.W)
        self.op = DotStep(self.manifold)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        return self.op(input, self.U, self.S, self.V)



def custom_sgd(loss_or_grads, params, learning_rate, manifolds = {}):
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps

    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    """
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    fixed_rank_tuple, fixed_rank_grads_tuple = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                                               if "fixed_rank" in param.name)))

    params, grads = list(zip(*tuple((param, grad) for (param, grad) in zip(params, grads)
                                                               if "fixed_rank" not in param.name)))
    params = [fixed_rank_tuple] + list(params)
    grads = [fixed_rank_grads_tuple] + list(grads)

    for param, grad in zip(params, grads):
        if param and isinstance(param, tuple) and "fixed_rank" in param[0].name:
            manifold = manifolds["fixed_rank"]
            param_updates = manifold.retr(param, grad, -learning_rate)
            for p, upd in zip(param, param_updates):
                updates[p] = upd
        else:
            updates[param] = param - learning_rate * grad

    return updates



def iterate_minibatches(X, y, batchsize):
        n_samples = X.shape[0]

        # Shuffle at the start of epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start in range(0, n_samples, batchsize):
            end = min(start + batchsize, n_samples)

            batch_idx = indices[start:end]

            yield X[batch_idx], y[batch_idx]
        if n_samples % batchsize != 0:
            batch_idx = indices[n_samples - n_samples % batchsize :]
            yield X[batch_idx], y[batch_idx]


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
    dense_1 = DotLayer2(input_layer,
                       num_units=20,
                       name = "fixed_rank")
    """
    dense_1 = lasagne.layers.DenseLayer(input_layer,
                       num_units=100, b=None,
                       name = "dense_1")
    """
    biased_1 = lasagne.layers.BiasLayer(dense_1)
    nonlin_1 = lasagne.layers.NonlinearityLayer(biased_1, lasagne.nonlinearities.rectify)

    #dense_1 = lasagne.layers.DenseLayer(input_layer,num_units=25,
    #                                   nonlinearity = lasagne.nonlinearities.sigmoid,
    #                                   name = "hidden_dense_layer")

    #ВЫХОДНОЙ полносвязный слой, который принимает на вход dense_1 и имеет 10 нейронов -по нейрону на цифру
    #нелинейность - softmax - чтобы вероятности всех цифр давали в сумме 1
    dense_output = lasagne.layers.DenseLayer(nonlin_1,num_units = 50,
                                            nonlinearity = lasagne.nonlinearities.softmax,
                                           name='output')

    #предсказание нейронки (theano-преобразование)
    y_predicted = lasagne.layers.get_output(dense_output)

    #все веса нейронки (shared-переменные)
    all_weights = lasagne.layers.get_all_params(dense_output)
    print(all_weights)

    #функция ошибки - средняя кроссэнтропия
    loss = lasagne.objectives.categorical_crossentropy(y_predicted,target_y).mean()
    l2_penalty = lasagne.regularization.regularize_layer_params(dense_1, lasagne.regularization.l2) * 1e-3
    loss += l2_penalty


    accuracy = lasagne.objectives.categorical_accuracy(y_predicted,target_y).mean()

    #сразу посчитать словарь обновлённых значений с шагом по градиенту, как раньше
    updates_sgd = lasagne.updates.sgd(loss, all_weights,learning_rate=0.01)
    #updates_sgd = custom_sgd(loss, all_weights, learning_rate=0.01, manifolds={"fixed_rank": dense_1.manifold})

    #функция, которая обучает сеть на 1 шаг и возвращащет значение функции потерь и точности
    theano.config.exception_verbosity = 'high'

    train_fun = theano.function([input_X,target_y],[loss,accuracy],updates=updates_sgd)
    accuracy_fun = theano.function([input_X,target_y],accuracy)

    num_epochs = 5 #количество проходов по данным

    batch_size = 50 #размер мини-батча

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,batch_size):
            inputs, targets = batch
            train_err_batch, train_acc_batch= train_fun(inputs, targets)
            train_err += train_err_batch
            train_acc += train_acc_batch
            train_batches += 1

        # And a full pass over the validation data:
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size):
            inputs, targets = batch
            val_acc += accuracy_fun(inputs, targets)
            val_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))

        print("  training loss (in-iteration):\t\t{:.6f}".format(train_err / train_batches))
        print("  train accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))