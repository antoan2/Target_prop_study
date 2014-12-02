import numpy as np
import theano
import sys
import theano.tensor as T
import utils.init_functions as ifunc
from theano import function
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import izip
import utils.learning_rules as ulru
import utils.plot_tools as uplot

def get_updates_lrs(d, current_epoch, lrs, lrs_i):
    updates = OrderedDict()
    for lr, lr_i in izip(lrs.values(), lrs_i):
        updates[lr] = lr_i/(1.+d*current_epoch)
    return updates

np.random.seed(53)

noise = .3
n_samples = 10000
batch_size = 100
n_epochs = 50
interval = 2*np.pi

dims = [1, 3, 3, 1]
lrs_f = np.asarray([0.005, 0.005, 0.005]).astype('float32')
d = 0.00005
momentum = 0.9

shareX = lambda x: theano.shared(np.asarray(x).astype('float32'))
f_to_approx = lambda x: 0.8*np.sin(x) + (np.random.rand(x.shape[0], 1)-.5)*noise
mse = lambda h, hh: T.sqr(h-hh).sum(axis=1).mean()

x_np = (np.random.rand(n_samples)-0.5)
x_np = x_np[..., None]
y_np = f_to_approx(x_np*interval)
x_th = theano.shared(x_np.astype('float32'))
y_th = theano.shared(y_np.astype('float32'))

W1, b1 = ifunc.init_W((dims[0], dims[1])), ifunc.init_b(dims[1])
W2, b2 = ifunc.init_W((dims[1], dims[2])), ifunc.init_b(dims[2])
W3, b3 = ifunc.init_W((dims[2], dims[3])), ifunc.init_b(dims[3])
params = [W1, b1, W2, b2, W3, b3]
lrs_i = list()
[lrs_i.extend((shareX(lrs_f[i]), shareX(lrs_f[i]))) for i in xrange(3)]
lrs = OrderedDict(izip(params, lrs_i))

f1 = lambda x: T.tanh(T.dot(x, W1) + b1)
f2 = lambda x: T.tanh(T.dot(x, W2) + b2)
f3 = lambda x: T.tanh(T.dot(x, W3) + b3)

x = T.fmatrix()
y = T.fmatrix()
index = T.lscalar()
current_epoch = T.fscalar()

h1 = f1(x)
h2 = f2(h1)
pred = f3(h2)

final_cost = mse(pred, y)

grads, updates = ulru.get_gradients(final_cost, params)
updates_lrs = get_updates_lrs(d, current_epoch, lrs, lrs_i)
updates.update(ulru.get_updates_sgd(params, lrs, grads))
#updates.update(ulru.get_updates_momentum(params, lrs, grads, momentum))
#updates.update(ulru.get_updates_adadelta(params, lrs, grads, 0.9))
#updates.update(ulru.get_updates_adagrad(params, lrs, grads))
#updates.update(ulru.get_updates_rmsprop(params, lrs, grads, 0.9))

one_step_train = function(inputs=[index],
                            updates=updates,
                            givens={
                                x:x_th[batch_size*index:batch_size*(index+1)],
                                y:y_th[batch_size*index:batch_size*(index+1)]})

update_lrs = function(inputs=[current_epoch],
                            updates=updates_lrs)
                            
get_cost = function(inputs=[index],
                            outputs=final_cost,
                            givens={
                                x:x_th[batch_size*index:batch_size*(index+1)],
                                y:y_th[batch_size*index:batch_size*(index+1)]})

get_variables = function(inputs=[x], outputs=[h1, h2])

get_pred = function(inputs=[x],
                    outputs=pred)

costs = []
n_batch_epoch = n_samples/batch_size
for epoch in xrange(n_epochs):
    print('\n')
    permut = np.random.permutation(n_samples)
    x_th = x_th[permut]
    y_th = y_th[permut]
    for batch in xrange(n_batch_epoch):
        one_step_train(batch)
        mean_cost = np.asarray([get_cost(i) for i in xrange(n_batch_epoch)]).mean()
        costs.append(mean_cost)
        sys.stdout.write('\repoch %d batch %d mean_cost %f'%(epoch, batch, mean_cost))
        sys.stdout.flush()
    update_lrs(epoch)

plt.figure('costs')
plt.plot(costs)

x_pred = np.linspace(-0.5, 0.5, 100).astype('float32')
x_pred = x_pred[..., None]
y_pred = get_pred(x_pred)
plt.figure()
plt.plot(x_np[::20, 0], y_np[::20, 0], '*')
plt.plot(x_pred, y_pred, '*')

h1, h2 = get_variables(x_pred)
uplot.plot_scatter(h1, 'layer 1')
uplot.plot_scatter(h2, 'layer 2')
plt.show()
