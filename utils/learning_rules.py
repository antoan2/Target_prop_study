import numpy as np
import theano
import sys
import theano.tensor as T
import utils.init_functions as ifunc
from theano import function
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import izip

def get_gradients(cost, params):
    grads = OrderedDict(izip(params, T.grad(cost, params, disconnected_inputs='ignore')))
    updates = OrderedDict()
    return grads, updates

def get_updates_lrs(d, current_epoch, lrs, lrs_i):
    updates = OrderedDict()
    for lr, lr_i in izip(lrs.values(), lrs_i):
        updates[lr] = lr_i/(1.+d*current_epoch)
    return updates

def get_updates_sgd(params, lrs, grads):
    return dict(izip(params, [param - lrs[param]*grads[param] for param in params]))

def get_updates_momentum(params, lrs, grads, momentum):
    updates = OrderedDict()
    for param in params:
        vel = theano.shared(param.get_value()*0.)
        updates[vel] = momentum*vel - lrs[param]*grads[param]
        inc = updates[vel]
        updates[param] = param + inc
    return updates

def get_updates_momentum_nestorov(params, lrs, grads, momentum):
    updates = OrderedDict()
    for param in params:
        vel = theano.shared(param.get_value()*0.)
        updates[vel] = momentum*vel - lrs[param]*grads[param]
        inc = updates[vel]
        inc = momentum*inc - lrs[param]*grads[param]
        updates[param] = param + inc
    return updates

def get_updates_adadelta(params, lrs, grads, decay):
    updates = OrderedDict()
    for param in params:
        mean_square_grad = theano.shared(param.get_value()*0.)
        mean_square_dx = theano.shared(param.get_value()*0.)

        new_mean_squared_grad = (decay*mean_square_grad + (1-decay)*T.sqr(grads[param]))

        epsilon = lrs[param]
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

        new_mean_square_dx = (decay*mean_square_dx + (1-decay)*T.sqr(delta_x_t))

        updates[mean_square_grad] = new_mean_squared_grad
        updates[mean_square_dx] = new_mean_square_dx
        updates[param] = param + delta_x_t
    
    return updates

def get_updates_adagrad(params, lrs, grads):
    updates = OrderedDict()
    for param in params:
        sum_square_grad = theano.shared(param.get_value()*0.)

        new_sum_squared_grad = sum_square_grad + T.sqr(grads[param])

        delta_x_t = -lrs[param]/T.sqrt(new_sum_squared_grad)*grads[param]

        updates[sum_square_grad] = new_sum_squared_grad
        updates[param] = param + delta_x_t

    return updates

def get_updates_rmsprop(params, lrs, grads, decay, max_scaling=1e5):
    updates = OrderedDict()
    for param in params:
        mean_square_grad = theano.shared(param.get_value()*0.)

        new_mean_squared_grad = decay*mean_square_grad + (1-decay)*T.sqr(grads[param])

        rms_grad_t = T.sqrt(new_mean_squared_grad)
        rms_grad_t = T.maximum(rms_grad_t, 1./max_scaling)
        delta_x_t = -lrs[param]*grads[param]/rms_grad_t

        updates[mean_square_grad] = new_mean_squared_grad
        updates[param] = param + delta_x_t

    return updates
