import theano
from utils import datasets
import theano.tensor as T
from theano import function
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse

# np.random.seed(453)

class mlp_target(object):

    def __init__(self, inputs, labels, layer_dimensions, learning_rate_t, learning_rates_f, learning_rates_g, weights=None, activation='tanh'):

        # initialisation of the composition paarameters
        self.layer_dimensions = layer_dimensions
        self.lr_t = learning_rate_t
        self.lr_f = learning_rates_f
        self.lr_g = learning_rates_g
        self.n_layers = len(layer_dimensions) - 1
        
        # initialisation of the layers
        self.layers = []
        # parameters for the mappings
        for i in xrange(self.n_layers):
            if not weights == None:
                current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1], weights=weights[i], activation=activation)
            else:
                current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1], activation=activation)
            self.layers.append(current_layer)

        # computation of the hidden layers
        self.variables = []
        self.variables.append(inputs)
        for i in xrange(self.n_layers):
            self.variables.append(self.layers[i].f(self.variables[i]))

        # classifier
        self.p_y_given_x = T.nnet.softmax(self.variables[self.n_layers])
        self.predict = T.argmax(self.p_y_given_x, axis=1)

        # costs
        self.cost = self.softmax_cross_entropy(labels)
        self.error = self.error(labels)

        # list to store the targets of the targets
        self.variables_targets = []
        # last layer target obtained by gradient descent on the layer before the softmax
        temp_target = self.variables[-2]-self.lr_t*T.grad(self.cost, self.variables[-2])
        self.variables_targets.append(temp_target)
        for i in xrange(self.n_layers-2, -1, -1):
            temp_target = self.variables[i]
            temp_target -= self.layers[i].g(self.variables[i+1])
            temp_target += self.layers[i].g(self.variables_targets[-1])
            self.variables_targets.append(temp_target)
        self.variables_targets.reverse()

        # for i in xrange(self.n_layers-1):
            # self.variables_targets[i] = self.variables[i] - 0.01*(self.variables_targets[i] - self.variables[i]) \
                                       # /(self.variables_targets[i] - self.variables[i]).norm(1)
        
        # list for the updates
        self.updates = []

        # computation of the target costs
        self.cost_targets = []
        for i in xrange(0, self.n_layers-1):
            self.cost_targets.append(self.mse(self.variables_targets[i+1], self.variables[i+1]))
            d_W, d_b = T.grad(self.cost_targets[-1], [self.layers[i].W, self.layers[i].b],
                                consider_constant=[self.variables[i], self.variables_targets[i+1]])
            self.updates.append((self.layers[i].W, self.layers[i].W-self.lr_f[i]*d_W))
            self.updates.append((self.layers[i].b, self.layers[i].b-self.lr_f[i]*d_b))

        # computation of the inverse mapping cost
        self.cost_inverses = []
        for i in xrange(1, self.n_layers-1):
            temp_var = self.variables_targets[i+1]
            temp_var_inversed = self.layers[i].f(self.layers[i].g(temp_var))
            self.cost_inverses.append(self.mse(temp_var_inversed, temp_var))
            d_V, d_c = T.grad(self.cost_inverses[-1], [self.layers[i].V, self.layers[i].c],
                                consider_constant=[self.variables_targets[i+1]])
            self.updates.append((self.layers[i].V, self.layers[i].V-self.lr_g[i-1]*d_V))
            self.updates.append((self.layers[i].c, self.layers[i].c-self.lr_g[i-1]*d_c))

        # computation of the gradients for the last layer
        d_W, d_b = T.grad(self.cost, [self.layers[-1].W, self.layers[-1].b], consider_constant=[self.variables[-2]])
        self.updates.append((self.layers[-1].W, self.layers[-1].W - self.lr_f[-1]*d_W))
        self.updates.append((self.layers[-1].b, self.layers[-1].b - self.lr_f[-1]*d_b))

    def softmax_cross_entropy(self, labels):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(labels.shape[0]), labels])

    def error(self, y):
        return T.mean(T.neq(self.predict, y))

    def mse(self, h, hh):
        return T.sqr(h-hh).sum(axis=1).mean()

class Layer(object):

    def __init__(self, nI, nO, weights=None, activation='tanh'):
        
        if not weights == None:
            self.W = theano.shared(weights.W)
            self.b = theano.shared(weights.b)
            self.V = theano.shared(weights.V)
            self.c = theano.shared(weights.c)
        else:
            self.W = self.init_parameters((nI, nO), 1., 0)
            self.b = self.init_parameters((nO,), 0, 0)
            # self.V = self.init_parameters((nO, nI), 1., 0)
            self.V = theano.shared(self.W.get_value().T)
            self.c = self.init_parameters((nI,), 0, 0)

        if activation=='tanh':
            f_act = lambda h: T.tanh(h)
        elif activation=='sigmoid':
            f_act = lambda h: T.nnet.sigmoid(h)

        self.f = lambda h: f_act(T.dot(h, self.W) + self.b)
        self.g = lambda h: f_act(T.dot(h, self.V) + self.c)

    def init_parameters(self, shape, std, mean):
        return theano.shared(np.random.standard_normal(shape).astype(dtype=theano.config.floatX)*std + mean)


def learning_target(learning_rate_t, learning_rates_f, learning_rates_g):
    # experience parameters
    layer_dimensions = [2, 3, 3, 3, 2]
    # fixed hyperparameters
    batch_size = 200
    n_batch_train = train_set_x.get_value().shape[0]/batch_size
    n_batch_train_compute = 100
    n_exp = 2000

    # initialization mlp
    x = T.fmatrix()
    y = T.ivector()
    index = T.lscalar()
    classif = mlp_target(x, y, layer_dimensions, learning_rate_t, learning_rates_f, learning_rates_g)

    one_step_train = function(inputs=[index],
                        outputs=classif.cost,
                        updates=classif.updates,
                        givens={
                        x:train_set_x[batch_size*index:batch_size*(index+1)],
                        y:train_set_y[batch_size*index:batch_size*(index+1)]})

    cost_train = function(inputs=[index],
                        outputs=classif.cost,
                        givens={
                            x:train_set_x[batch_size*index:batch_size*(index+1)],
                            y:train_set_y[batch_size*index:batch_size*(index+1)]})
 
    error_train = function(inputs=[index],
                        outputs=classif.error,
                        givens={
                            x:train_set_x[batch_size*index:batch_size*(index+1)],
                            y:train_set_y[batch_size*index:batch_size*(index+1)]})

    best_error = 1
    sys.stdout.write('batch %d'%0)
    sys.stdout.flush()
    last_cost = np.inf
    for current_batch in xrange(n_exp):
     
        if current_batch%n_batch_train == 0:
            epoch = current_batch/n_batch_train
     
        index_tab = current_batch
        current_batch = current_batch%n_batch_train 
        mean_cost = np.asarray([cost_train(i) for i in xrange(n_batch_train_compute)]).mean()
        if last_cost == mean_cost:
            sys.stdout.write('\nmean_error %f\n'%mean_error)
            sys.stdout.flush()
            return np.inf
        mean_error = np.asarray([error_train(i) for i in xrange(n_batch_train_compute)]).mean()
        if mean_error < best_error:
            best_error = mean_error
        one_step_train(current_batch)
        
        sys.stdout.write('\rbatch %d mean_cost %f mean_error %f'%(index_tab, mean_cost, mean_error))
        sys.stdout.flush()
        last_cost = mean_cost
    sys.stdout.write('\nmean_error : %f\n'%mean_error)
    sys.stdout.flush()
    return mean_error

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='run experiments')
    parser.add_argument('save_name', type=str)
    args = parser.parse_args()

    # initialisation dataset
    dataset_file = 'datasets/ellipse_50000.pkl'
    [(train_set_x, train_set_y), \
           (valid_set_x, valid_set_y), \
           (test_set_x, test_set_y)] = datasets.load_dataset(dataset_file)

    tab_lr_t = 0
    tab_lr_f = []
    tab_lr_g = []
    tab_error = []

    best_lr_t = []
    best_lr_f = []
    best_lr_g = []
    best_error = np.inf

    interval_t = (-2, 2)
    intervals_f = ((-4, 2), (-4, 2), (-4, 2), (-4, 1))
    intervals_g = ((-9, -2), (-9, -2))
    learning_rates_f = np.zeros(4)
    learning_rates_g = np.zeros(2)
    for i in range(100):
        print('\nexperiment '+str(i))
        learning_rate_t = 10**(interval_t[0]+(interval_t[1]-interval_t[0])*np.random.rand())
        for i in range(len(learning_rates_f)):
            learning_rates_f[i] = 10**(intervals_f[i][0]+(intervals_f[i][1]-intervals_f[i][0])*np.random.rand())
        learning_rates_f = learning_rates_f.astype(theano.config.floatX)
        for i in range(len(learning_rates_g)):
            learning_rates_g[i] = 10**(intervals_g[i][0]+(intervals_g[i][1]-intervals_g[i][0])*np.random.rand())
        learning_rates_g = learning_rates_g.astype(theano.config.floatX)
        print(learning_rate_t)
        print(learning_rates_f)
        print(learning_rates_g)
        current_error = learning_target(learning_rate_t, learning_rates_f, learning_rates_g)
        tab_lr_t.append(learning_rate_t)
        tab_lr_f.append(learning_rates_f)
        tab_lr_g.append(learning_rates_g)
        tab_error.append(current_error)
        if current_error < best_error:
            best_lr_t = learning_rate_t
            best_lr_f = learning_rates_f
            best_lr_g = learning_rates_g
            best_error = current_error

    print('\nbest_parameters :')
    print('best_error : '+str(best_error))
    print('best_lr_t : '+str(best_lr_t))
    print('best_lr_f : '+str(best_lr_f))
    print('best_lr_g : '+str(best_lr_g))

    tab_error = np.asarray(tab_error)
    tab_lr_f = np.asarray(tab_lr_f)
    tab_lr_g = np.asarray(tab_lr_g)

    ind = tab_error.argsort()
    tab_error = tab_error[ind]
    tab_lr_f = tab_lr_f[ind]
    tab_lr_g = tab_lr_g[ind]
    params = dict()
    params['error'] = tab_error
    params['lr_f'] = tab_lr_f
    params['lr_g'] = tab_lr_g

    print('saving')
    cPickle.dump(params, open(args.save_name, 'w'))
