import theano
from utils import datasets
import theano.tensor as T
from theano import function
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(453)

class simple_composition(object):

    def __init__(self, inputs, labels, layer_dimensions, learning_rates):

        # initialisation of the composition paarameters
        self.layer_dimensions = layer_dimensions
        self.learning_rates = learning_rates
        self.n_layers = len(layer_dimensions) - 1
        
        # initialisation of the layers
        self.layers = []
        # parameters for the mappings
        for i in xrange(self.n_layers):
           current_layer = Layer(layer_dimensions[i], layer_dimensions[i+1])
           self.layers.append(current_layer)

        # computation of the hidden layers
        self.variables = []
        self.variables.append(inputs)
        for i in xrange(self.n_layers):
            self.variables.append(self.layers[i].f(self.variables[i]))

        self.p_y_given_x = T.nnet.softmax(self.variables[self.n_layers])
        self.predict = T.argmax(self.p_y_given_x, axis=1)
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

    def __init__(self, nI, nO):
        
        self.W = self.init_parameters((nI, nO), 1., 0)
        self.b = self.init_parameters((nO,), 0, 0)
        self.f = lambda h: T.tanh(T.dot(h, self.W) + self.b)

    def init_parameters(self, shape, std, mean):
        return theano.shared(np.random.standard_normal(shape).astype(dtype=theano.config.floatX)*std + mean)

dataset_file = 'datasets/ellipse_50000.pkl'
[(train_set_x, train_set_y), \
       (valid_set_x, valid_set_y), \
       (test_set_x, test_set_y)] = datasets.load_dataset(dataset_file)

x = T.fmatrix()
y = T.ivector()
index = T.lscalar()

layer_dimensions = [2, 3, 3, 2]
learning_rates = [0.05, 0.05, 0.05]
batch_size = 200
n_batch_train = train_set_x.get_value().shape[0]/batch_size
n_batch_train_compute = 100
n_exp = 50000
d = 0.05

classif = simple_composition(x, y, layer_dimensions, learning_rates)
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

fprop = function(inputs=[x],
                    outputs=classif.variables)

x_c = train_set_x.get_value()
x_c = x_c[::20, ...]
y_c = train_set_y.eval().astype('bool')
y_c = y_c[::20, ...]
x_1 = x_c[y_c]
x_2 = x_c[-y_c]
print(x_1.shape)
print(x_2.shape)
bool_plot = False
cost_tab = np.zeros(n_exp)
error_tab = np.zeros(n_exp)
for current_batch in xrange(n_exp):
    if current_batch%n_batch_train == 0:
        epoch = current_batch/n_batch_train
        for i in xrange(len(learning_rates)):
            classif.learning_rates[i] = learning_rates[i]/(1+d*epoch)
        print(classif.learning_rates)
    index_tab = current_batch
    current_batch = current_batch%n_batch_train 
    mean_cost = np.asarray([cost_train(i) for i in xrange(n_batch_train_compute)]).mean()
    mean_error = np.asarray([error_train(i) for i in xrange(n_batch_train_compute)]).mean()
    cost, predict = one_step_train(current_batch)
    cost_tab[index_tab] = mean_cost
    error_tab[index_tab] = mean_error
    print(index_tab)
    print(classif.learning_rates)
    print(mean_cost)
    print(mean_error)
    if bool_plot:
        if current_batch%1000 == 0:
            x_to_plot, pred, y_to_plot = predict_train(current_batch)
            x_to_plot = x_to_plot[pred.astype('bool')]
            plt.scatter(x_to_plot[:, 0], x_to_plot[:, 1])
            plt.axis([-1, 1, -1, 1])
            plt.show()

if bool_plot:
    plt.plot(cost_tab)
    plt.show()
    plt.plot(error_tab)
    plt.show()

    hs = []
    c = ['r', 'b']
    m = ['o', '^']
    hs.append(fprop(x_1))
    hs.append(fprop(x_2))
    for i in xrange(len(hs[0])):
        print(i)
        fig = plt.figure()
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
