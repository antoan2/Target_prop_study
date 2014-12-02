import numpy as np
import utils.plot_tools as uplot
import utils.init_functions as ifunc
import utils.datasets as udat
import utils.save_tools as stool
from theano.compat.python2x import OrderedDict
import sys
from theano import function
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

#np.random.seed(53)

mse = lambda h, hh : T.sqr(h - hh).sum(axis=1).mean()
#f_to_approx = lambda x : 0.8*np.sin(x+np.pi) + noise*np.random.rand(x.shape[0], 1)

n_samples = 10000
n_epochs = 100
interval = 2*np.pi
noise = 0.1
batch_size = 100

dim = [1, 3, 3, 1]
d = 0.0
lrs_f = np.asarray([.5, .5, .5], dtype='float32')
params = []
lr_t = np.asarray(.5, dtype='float32')
lr_g = np.asarray(.08, dtype='float32')
lrs = []

x_train_set, y_train_set, f_to_approx = udat.generate_sinus(n_samples, interval, noise)

W1 = ifunc.init_W((dim[0], dim[1]))
b1 = ifunc.init_b(dim[1])
params.extend((W1, b1))
lrs.extend((theano.shared(lrs_f[0]), theano.shared(lrs_f[0])))
W2 = ifunc.init_W((dim[1], dim[2]))
b2 = ifunc.init_b(dim[2])
V2 = ifunc.init_W((dim[2], dim[1]))
c2 = ifunc.init_b(dim[1])
params.extend((W2, b2))
lrs.extend((theano.shared(lrs_f[1]), theano.shared(lrs_f[1])))
params.extend((V2, c2))
lrs.extend((theano.shared(lr_g), theano.shared(lr_g)))
W3 = ifunc.init_W((dim[2], dim[3]))
b3 = ifunc.init_b(dim[3])
params.extend((W3, b3))
lrs.extend((theano.shared(lrs_f[2]), theano.shared(lrs_f[2])))
lrs_i = lrs

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

#norm_cost = T.grad(cost, h2).norm(2, axis=1).mean()
hh2 = h2 - lr_t*T.grad(cost, h2)
#norm_2 = (hh2 - h2).norm(2, axis=1).mean()
#hh2 = h2 - 0.1*(hh2 - h2)/norm_2*norm_cost
hh1 = h1 + g2(hh2) - g2(h2)
#norm_1 = (hh1 - h1).norm(2, axis=1).mean()
#hh1 = h1 - 0.1*(hh1 - h1)/norm_1*norm_cost

get_targets = function(inputs=[x, y],
                            outputs=[hh1, hh2])

cost_target_1 = mse(h1, hh1)
cost_target_2 = mse(h2, hh2)
cost_inverse = mse(f2(g2(hh2)), hh2)

grads = []
d_W, d_b = T.grad(cost_target_1, [W1, b1], consider_constant=[hh1])
grads.extend((d_W, d_b))
d_W, d_b = T.grad(cost_target_2, [W2, b2], consider_constant=[hh2, h1])
grads.extend((d_W, d_b))
d_V, d_c = T.grad(cost_inverse, [V2, c2], consider_constant=[hh2])
grads.extend((d_V, d_c))
d_W, d_b = T.grad(cost, [W3, b3], consider_constant=[h2])
grads.extend((d_W, d_b))

updates = [(param, param-step*grad) for (param, step, grad) in zip(params, lrs, grads)]
updates_lrs = [(lr, lr/(1.+d*current_epoch)) for lr in lrs]

lrs_update = function(inputs=[current_epoch],
                        updates=updates_lrs)
one_step_train = function(inputs=[index],
                            outputs = cost,
                            updates = updates,
                            givens = {
                                x:x_train_set[batch_size*index:batch_size*(index+1)],
                                y:y_train_set[batch_size*index:batch_size*(index+1)]})

get_cost = function(inputs=[index],
                            outputs = [cost, cost_target_1, cost_target_2, cost_inverse],
                            givens = {
                                x:x_train_set[batch_size*index:batch_size*(index+1)],
                                y:y_train_set[batch_size*index:batch_size*(index+1)]})

n_batch_epoch = n_samples/batch_size
costs = np.zeros((n_epochs*n_batch_epoch, 4))
for epoch in xrange(n_epochs):
    print('\n')
    ind = np.random.permutation(n_samples)
    x_train_set = x_train_set[ind]
    y_train_set = y_train_set[ind]
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

x_plot = x_train_set.eval()
y_plot = y_train_set.eval()
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

stool.save_target_3([param.get_value() for param in params], costs, 'Curves_sinus_approx/target_sgd_2.pkl')

plt.show()
