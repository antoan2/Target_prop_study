import theano
from utils import datasets
import theano.tensor as T
from theano import function
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(453)

class mlp_backprop(object):

    def __init__(self, inputs, labels, layer_dimensions, learning_rates, weights=None, activation='tanh'):

        # initialisation of the composition paarameters
        self.layer_dimensions = layer_dimensions
        self.learning_rates = learning_rates
        self.n_layers = len(layer_dimensions) - 1
        
        # initialisation of the layers
        self.layers = []
        # parameters for the mappings
        for i in xrange(self.n_layers-1):
            if not weights == None:
                current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1], weights=weights[i], activation=activation)
            else:
                current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1], activation=activation)
            self.layers.append(current_layer)
        if not weights == None:
            current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1], weights=weights[i], activation='softmax')
        else:
            current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1], activation='softmax')
        self.layers.append(current_layer)

        # computation of the hidden layers
        self.variables = []
        self.variables.append(inputs)
        for i in xrange(self.n_layers):
            self.variables.append(self.layers[i].f(self.variables[i]))

        # classifier
        self.p_y_given_x = self.variables[-1]
        self.predict = T.argmax(self.p_y_given_x, axis=1)

        # costs
        self.cost = self.softmax_cross_entropy(labels)
        self.error = self.error(labels)
        
        # computation of the gradients
        self.updates = []
        for i in xrange(self.n_layers):
            d_W, d_b = T.grad(self.cost, [self.layers[i].W, self.layers[i].b], consider_constant=[self.variables[i]])
            self.updates.append((self.layers[i].W, self.layers[i].W - self.learning_rates[i]*d_W))
            self.updates.append((self.layers[i].b, self.layers[i].b - self.learning_rates[i]*d_b))

    def softmax_cross_entropy(self, labels):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(labels.shape[0]), labels])

    def error(self, y):
        return T.mean(T.neq(self.predict, y))

class Layer(object):

    def __init__(self, nI, nO, weights=None, activation='tanh'):
        
        if not weights == None:
            self.W = theano.shared(weights.W)
            self.b = theano.shared(weights.b)
        else:
            self.W = self.init_parameters((nI, nO), 1., 0)
            self.b = self.init_parameters((nO,), 0, 0)

        if activation=='tanh':
            f_act = lambda h: T.tanh(h)
        elif activation=='sigmoid':
            f_act = lambda h: T.nnet.sigmoid(h)
        elif activation=='softmax':
            f_act = lambda h: T.nnet.softmax(h)

        self.f = lambda h: f_act(T.dot(h, self.W) + self.b)

    def init_parameters(self, shape, std, mean):
        return theano.shared(np.random.standard_normal(shape).astype(dtype=theano.config.floatX)*std + mean)

# initialisation dataset
dataset_file = 'datasets/circles_50000.pkl'
[(train_set_x, train_set_y), \
       (valid_set_x, valid_set_y), \
       (test_set_x, test_set_y)] = datasets.load_dataset(dataset_file)

# experience parameters
layer_dimensions = [2, 3, 3, 3, 2]
learning_rates = [0.1, 0.1, 0.1, 0.1, 0.1]
batch_size = 200
n_batch_train = train_set_x.get_value().shape[0]/batch_size
n_batch_train_compute = 100
n_exp = 800
d = 0.5

# initialization mlp
x = T.fmatrix()
y = T.ivector()
index = T.lscalar()
classif = mlp_backprop(x, y, layer_dimensions, learning_rates)

one_step_train = function(inputs=[index],
                    outputs=[classif.cost, classif.p_y_given_x],
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

predict_train = function(inputs=[index],
                    outputs=[x, classif.predict, y],
                    givens={
                        x:train_set_x[batch_size*index:batch_size*(index+1)],
                        y:train_set_y[batch_size*index:batch_size*(index+1)]})

# learning parameters
bool_plot_final = True
bool_plot_learning = False
bool_save = True

# initialization learning variables
cost_tab = np.zeros(n_exp)
error_tab = np.zeros(n_exp)
best_error = 1
for current_batch in xrange(n_exp):

    if current_batch%n_batch_train == 0:
        epoch = current_batch/n_batch_train
        for i in xrange(len(learning_rates)):
            classif.learning_rates[i] = learning_rates[i]/(1+d*epoch)

    index_tab = current_batch
    current_batch = current_batch%n_batch_train 
    mean_cost = np.asarray([cost_train(i) for i in xrange(n_batch_train_compute)]).mean()
    mean_error = np.asarray([error_train(i) for i in xrange(n_batch_train_compute)]).mean()

    if mean_error < best_error:
        print(mean_error)
        classif_best = classif

    cost, predict = one_step_train(current_batch)
    cost_tab[index_tab] = mean_cost
    error_tab[index_tab] = mean_error
    print(index_tab)
    print(mean_cost)
    print(mean_error)

    if bool_plot_learning:
        if current_batch%5000 == 0:
            x_to_plot, pred, y_to_plot = predict_train(current_batch)
            x_to_plot = x_to_plot[pred.astype('bool')]
            plt.scatter(x_to_plot[:, 0], x_to_plot[:, 1])
            plt.axis([-1, 1, -1, 1])
            plt.show()

if bool_save:
    weights = []
    for i in xrange(classif.n_layers):
        weights.append((classif_best.layers[i].W.get_value(), classif_best.layers[i].b.get_value()))
    file_to_save = open('temp_save.pkl', 'w')
    cPickle.dump(weights, file_to_save)
    file_to_save.close()

fprop = function(inputs=[x],
                    outputs=classif_best.variables)

x_c = train_set_x.get_value()
x_c = x_c[::20, ...]
y_c = train_set_y.eval().astype('bool')
y_c = y_c[::20, ...]
x_1 = x_c[y_c]
x_2 = x_c[-y_c]
print(x_1.shape)
print(x_2.shape)

if bool_plot_final:
    fig = plt.figure()
    fig.suptitle('cost and error - backprop')
    ax = fig.add_subplot(121)
    ax.plot(cost_tab)
    ax.set_title('cost function')
    ax.set_xlabel('batches seen')
    ax.set_ylabel('cost')
    ax = fig.add_subplot(122)
    ax.plot(error_tab)
    ax.set_title('error function')
    ax.set_xlabel('batches seen')
    ax.set_ylabel('error')
    plt.show()

    hs = []
    c = ['r', 'b']
    m = ['o', '^']
    hs.append(fprop(x_1))
    hs.append(fprop(x_2))
    for i in xrange(len(hs[0])-1):
        print(i)
        fig = plt.figure()
        fig.suptitle('layer '+str(i)+' - backprop')
        if (classif.n_layers>i) and (i>0):
            ax = fig.add_subplot(111, projection='3d')
            for p in xrange(2):
                hs_c = hs[p][i]
                c_c = c[p]
                m_c = m[p]
                print(hs_c.shape)
                ax.scatter3D(hs_c[:,0], hs_c[:,1], hs_c[:,2], c=c_c, marker=m_c)
        else:
            ax = fig.add_subplot(111, projection='3d')
            for p in xrange(2):
                hs_c = hs[p][i]
                c_c = c[p]
                m_c = m[p]
                print(hs_c.shape)
                ax.scatter(hs_c[:,0], hs_c[:,1], c=c_c, marker=m_c)
            
plt.show()
