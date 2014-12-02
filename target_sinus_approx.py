import numpy as np
import utils.plot_tools as uplot
import utils.init_functions as ifunc
import utils.learning_rules as ulru
import utils.datasets as udat
import utils.save_tools as stool
from collections import OrderedDict
from itertools import izip
import sys
from theano import function
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

def get_updates_lrs(d, current_epoch, lrs, lrs_i):
    updates = OrderedDict()
    for lr, lr_i in izip(lrs.values(), lrs_i):
        updates[lr] = lr_i/(1.+d*current_epoch)
    return updates

np.random.seed(53)

shareX = lambda x: theano.shared(np.asarray(x).astype('float32'))
mse = lambda h, hh : T.sqr(h - hh).sum(axis=1).mean()
f_to_approx = lambda x : 0.8*np.sin(x) + noise*(np.random.rand(x.shape[0], 1)-.5)

n_samples = 10000
n_epochs = 100
interval = 2*np.pi
noise = 0.1
batch_size = 100

dim = [1, 3, 3, 1]

#d = 0.0
#lrs_f = np.asarray([.5, .5, .5], dtype='float32')
#lr_t = np.asarray(.5, dtype='float32')
#lr_g = np.asarray(.5, dtype='float32')
# adagrad and rmsprop setting
d = 0.00005
lrs_f = np.asarray([.01, .01, .01], dtype='float32')
lr_t = np.asarray(.01, dtype='float32')
lr_g = np.asarray(.01, dtype='float32')

x_np = (np.random.rand(n_samples)-0.5)
x_np = x_np[..., None]
y_np = f_to_approx(x_np*interval)
x_th = theano.shared(x_np.astype('float32'))
y_th = theano.shared(y_np.astype('float32'))

W1 = ifunc.init_W((dim[0], dim[1]))
b1 = ifunc.init_b(dim[1])
W2 = ifunc.init_W((dim[1], dim[2]))
b2 = ifunc.init_b(dim[2])
V2 = ifunc.init_W((dim[2], dim[1]))
c2 = ifunc.init_b(dim[1])
W3 = ifunc.init_W((dim[2], dim[3]))
b3 = ifunc.init_b(dim[3])
params = [W1, b1, W2, b2, W3, b3, V2, c2]
lrs_i = list()
[lrs_i.extend((shareX(lrs_f[i]), shareX(lrs_f[i]))) for i in xrange(3)]
lrs_i.extend((shareX(lr_g), shareX(lr_g)))
lrs = OrderedDict(izip(params, lrs_i))

f1 = lambda h : T.tanh(T.dot(h, W1) + b1)
f2 = lambda h : T.tanh(T.dot(h, W2) + b2)
g2 = lambda h : T.tanh(T.dot(h, V2) + c2)
f3 = lambda h : T.tanh(T.dot(h, W3) + b3)

x = T.fmatrix()
y = T.fmatrix()
index = T.lscalar()
current_epoch = T.fscalar()
h1 = f1(x)
h2 = f2(h1)
predictions = f3(h2)

get_predictions = function(inputs=[x],
                            outputs=predictions)
get_variables = function(inputs=[x],
                            outputs=[h1, h2])

cost = mse(predictions, y)

hh2 = h2 - lr_t*T.grad(cost, h2)
hh1 = h1 + g2(hh2) - g2(h2)

get_targets = function(inputs=[x, y],
                            outputs=[hh1, hh2])

cost_target_1 = mse(h1, hh1)
cost_target_2 = mse(h2, hh2)
cost_inverse = mse(f2(g2(hh2)), hh2)

grads = OrderedDict()
grads[W1], grads[b1] = T.grad(cost_target_1, [W1, b1], consider_constant=[hh1])
grads[W2], grads[b2] = T.grad(cost_target_2, [W2, b2], consider_constant=[hh2, h1])
grads[W3], grads[b3] = T.grad(cost, [W3, b3], consider_constant=[h2])
grads[V2], grads[c2] = T.grad(cost_inverse, [V2, c2], consider_constant=[hh2])

updates_lrs = get_updates_lrs(d, current_epoch, lrs, lrs_i)
#updates = ulru.get_updates_sgd(params, lrs, grads)
#updates = ulru.get_updates_momentum(params, lrs, grads, 0.7)
#updates = ulru.get_updates_adadelta(params, lrs, grads, 0.7)
updates = ulru.get_updates_adagrad(params, lrs, grads)
#updates = ulru.get_updates_rmsprop(params, lrs, grads, 0.7)

lrs_update = function(inputs=[current_epoch],
                        updates=updates_lrs)
one_step_train = function(inputs=[index],
                            outputs = cost,
                            updates = updates,
                            givens = {
                                x:x_th[batch_size*index:batch_size*(index+1)],
                                y:y_th[batch_size*index:batch_size*(index+1)]})

get_cost = function(inputs=[index],
                            outputs = [cost, cost_target_1, cost_target_2, cost_inverse],
                            givens = {
                                x:x_th[batch_size*index:batch_size*(index+1)],
                                y:y_th[batch_size*index:batch_size*(index+1)]})

n_batch_epoch = n_samples/batch_size
costs = np.zeros((n_epochs*n_batch_epoch, 4))
for epoch in xrange(n_epochs):
    print('\n')
    ind = np.random.permutation(n_samples)
    x_th = x_th[ind]
    y_th = y_th[ind]
    for batch in xrange(n_batch_epoch):
        cost = one_step_train(batch)
        current_costs = np.zeros(4)
        n_cost = n_batch_epoch/10
        for i in xrange(n_cost):
            current_costs += np.asarray(get_cost(i))
        current_costs /= n_cost
        costs[epoch*n_batch_epoch+batch, :] = current_costs
        sys.stdout.write('\repoch %d batch %d mean_cost %f'%(epoch, batch, current_costs[0]))
        sys.stdout.flush()
    lrs_update(epoch)

costs = np.asarray(costs)
plt.figure('final_cost')
plt.plot(costs[:, 0])
plt.figure('cost_target_1')
plt.plot(costs[:, 1])
plt.figure('cost_target_2')
plt.plot(costs[:, 2])
plt.figure('cost_inverse')
plt.plot(costs[:, 3])

x_plot = x_th.eval()
y_plot = y_th.eval()
plt.figure()
plt.plot(x_plot[::20], y_plot[::20], 'o')

x_to_pred = np.linspace(-.5, .5, 100)
x_to_pred = x_to_pred[..., None].astype('float32')
y_to_pred = get_predictions(x_to_pred)
plt.plot(x_to_pred, y_to_pred, '*')

variables = get_variables(x_to_pred)
y_for = f_to_approx(x_to_pred)
targets = get_targets(x_to_pred, y_for.astype('float32'))
uplot.plot_scatter_targets(variables[0], targets[0], 'layer 1')
uplot.plot_scatter_targets(variables[1], targets[1], 'layer 2')

plt.show()
