import numpy as np
import sys
from theano import function
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import utils.plot_tools as uplot

np.random.seed(53)

def init_parameters_W(shape):
    std = 2.*(6./(shape[0] + shape[1]))**(0.5)
    return theano.shared(np.random.standard_normal(shape).astype(dtype=theano.config.floatX)*std)

def init_parameters_b(shape):
    return theano.shared(np.zeros(shape).astype(dtype=theano.config.floatX))

mse = lambda h, hh : T.sqr(h - hh).mean()

n_samples = 10000
n_epochs = 30
interval = 2*np.pi
noise = 0.1
batch_size = 100

dim = [1, 50, 50, 1]
d = 0.000001
lrs_f = np.asarray([.01, .01, .01], dtype='float32')
params = []
lrs = []
lrs_i = []

x_train_set_np = interval*np.random.rand(n_samples)
x_train_set_np = x_train_set_np[..., None]
y_train_set_np = 0.8*np.sin(x_train_set_np) + noise*np.random.rand(n_samples, 1)
#y_train_set_np = 0.1*(x_train_set_np-np.pi)**2 + noise*np.random.rand(n_samples, 1)
y_train_set_np = y_train_set_np
x_train_set = theano.shared(x_train_set_np.astype('float32'))
y_train_set = theano.shared(y_train_set_np.astype('float32'))

W1 = init_parameters_W((dim[0], dim[1]))
b1 = init_parameters_b(dim[1])
params.extend((W1, b1))
lrs.extend((theano.shared(lrs_f[0]), theano.shared(lrs_f[0])))
W2 = init_parameters_W((dim[1], dim[2]))
b2 = init_parameters_b(dim[2])
params.extend((W2, b2))
lrs.extend((theano.shared(lrs_f[1]), theano.shared(lrs_f[1])))
W3 = init_parameters_W((dim[2], dim[3]))
b3 = init_parameters_b(dim[3])
params.extend((W3, b3))
lrs.extend((theano.shared(lrs_f[2]), theano.shared(lrs_f[2])))
lrs_i = lrs

f1 = lambda h : T.tanh(T.dot(h, W1) + b1)
f2 = lambda h : T.tanh(T.dot(h, W2) + b2)
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
grads = [T.grad(cost, param) for param in params]
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
                            outputs = cost,
                            givens = {
                                x:x_train_set[batch_size*index:batch_size*(index+1)],
                                y:y_train_set[batch_size*index:batch_size*(index+1)]})

costs = []
n_batch_epoch = n_samples/batch_size
for epoch in xrange(n_epochs):
    print('\n')
    for batch in xrange(n_batch_epoch):
        cost = one_step_train(batch)
        mean_cost = np.asarray([get_cost(i) for i in xrange(n_batch_epoch/10)]).mean()
        costs.append(mean_cost)
        sys.stdout.write('\repoch %d batch %d mean_cost %f'%(epoch, batch, mean_cost))
        sys.stdout.flush()
    lrs_update(epoch)

costs = np.asarray(costs)
plt.figure()
plt.plot(costs)

x_plot = x_train_set.get_value()
y_plot = y_train_set.get_value()
plt.figure()
plt.plot(x_plot[::20], y_plot[::20], 'o')

x_to_pred = np.linspace(0, interval, 100)
x_to_pred = x_to_pred[..., None].astype('float32')
y_to_pred = get_predictions(x_to_pred)
plt.plot(x_to_pred, y_to_pred, '*')

h1, h2 = get_variables(x_to_pred)
uplot.plot_scatter(h1, 'layer 1')
uplot.plot_scatter(h2, 'layer 2')
plt.show()

