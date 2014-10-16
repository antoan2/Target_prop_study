from theano import function
from utils import datasets
import numpy as np
import time
import os
import pickle
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import argparse

class Saving(object):

    def __init__(self, channel_dict=[], n_exp=0):

        self.saving_dict = dict()
        for channel in channel_dict:
            self.saving_dict[channel] = np.zeros((n_exp,))
        
    def add_channel(self, channel):
        self.saving_dict[channel] = dict()
    
    def add_to_channel(self, channel, batch, value):
        self.saving_dict[channel][batch] = value

    def plot(self, channel, ax):
        ax.set_title(channel)
        ax.plot(self.saving_dict[channel])

    def load_pickle(self, pickle_name):
        self.saving_dict = pickle.load(open(pickle_name, 'rb'))

class target_inverse(object):

    def __init__(self, inputs, labels):

        # parameters 0 to 1 layer
        self.W1 = self.init_parameters((nX, nH1), 0.01, 0)
        self.b1 = self.init_parameters((nH1,), 0, 0)

        # parameters 1 to 2 layer
        self.W2 = self.init_parameters((nH1, nH2), 0.01, 0)
        self.b2 = self.init_parameters((nH2,), 0, 0)
        self.V2 = theano.shared(self.W2.get_value().T)
        self.c2 = self.init_parameters((nH1,), 0, 0)

        # parameters for cost layer
        self.W3 = self.init_parameters((nH2, nY), 0.01, 0)
        self.b3 = self.init_parameters((nY,), 0, 0)
        
        # definition of the mappings
        self.f1 = lambda input : T.tanh(T.dot(input, self.W1) + self.b1)
        self.f2 = lambda h1 : T.tanh(T.dot(h1, self.W2) + self.b2)
        self.g2 = lambda h2 : T.tanh(T.dot(h2, self.V2) + self.c2)
        self.f3 = lambda h2 : T.nnet.softmax(T.dot(self.h2, self.W3) + self.b3)

        # computation of the forward propagation
        self.h1 = self.f1(inputs)
        self.h2 = self.f2(self.h1) 
        self.p_y_given_x = self.f3(self.h2)
        self.final_cost = self.softmax_cross_entropy(labels)

        # computation of the two targets
        self.hh2 = self.h2 - lr_t_2*T.grad(self.softmax_cross_entropy(labels), self.h2)
        self.hh1 = self.h1 \
                        + self.g2(self.hh2) \
                        - self.g2(self.h2)

        # costs between targets and hidden variables
        self.cost_target_1 = self.mse(self.hh1, self.h1)
        self.cost_target_2 = self.mse(self.hh2, self.h2)

        # costs for inverse mapping
        self.cost_inverse_mapping_1 = self.mse(self.f2(self.g2(self.hh2)), self.hh2)

        # computing the gradients
        d_W3, d_b3 = T.grad(self.final_cost, [self.W3, self.b3], consider_constant=[self.h2])
        d_W2, d_b2 = T.grad(self.cost_target_2, [self.W2, self.b2], consider_constant=[self.hh2, self.h1])
        d_W1, d_b1 = T.grad(self.cost_target_1, [self.W1, self.b1], consider_constant=[self.hh1])
        d_V2, d_c2 = T.grad(self.cost_inverse_mapping_1, [self.V2, self.c2], consider_constant=[self.hh2])

        # update rule
        self.updates = [(self.W1, self.W1 - lr_f_1*d_W1), \
                    (self.b1, self.b1 - lr_f_1*d_b1), \
                    (self.W2, self.W2 - lr_f_2*d_W2), \
                    (self.b2, self.b2 - lr_f_2*d_b2), \
                    (self.W3, self.W3 - lr_f_3*d_W3), \
                    (self.b3, self.b3 - lr_f_3*d_b3), \
                    (self.V2, self.V2 - lr_b_2*d_V2), \
                    (self.c2, self.c2 - lr_b_2*d_c2)]

        # computation of the predictions
        self.predict = T.argmax(self.p_y_given_x, axis=1)

    def softmax_cross_entropy(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def error(self, y):
        return T.mean(T.neq(self.predict, y))

    def mse(self, hh, h):
        return T.sqr(hh-h).sum(axis=1).mean()

    def init_parameters(self, shape, std, mean):
        return theano.shared(np.random.standard_normal(shape).astype(dtype=theano.config.floatX)*std + mean)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='run an taget inverse mapping') 
    parser.add_argument('--plot', dest='plot_option',
                        help='if the script will plot some results')
    parser.add_argument('--save', dest='save_option',
                        help='if the script will save some results')
    args = parser.parse_args()
    print(args)

    dataset_file = os.environ['MNIST_LOCATION']
    [(train_set_x, train_set_y),\
    (valid_set_x, valid_set_y),\
    (test_set_x, test_set_y)] = datasets.load_dataset(dataset_file)

    x = T.fmatrix()
    y = T.ivector()
    index = T.lscalar()

    n_exp = 500
    plot_option = args.plot_option
    save_option = args.save_option
    nX, nH1, nH2, nY = 784, 1500, 1500, 10
    lr_t_2 = 5.
    lr_f_1, lr_f_2, lr_f_3, lr_b_2 = 5., 0.005, 0.15, 0.00001
    batch_size = 100
    model = target_inverse(x, y)

    one_step_train = function(inputs=[index],
                            outputs=[model.final_cost, model.cost_target_1, model.cost_target_2, model.cost_inverse_mapping_1, model.h1, model.h2],
                            updates=model.updates,
                            givens={
                                x:train_set_x[batch_size*index:(index+1)*batch_size],
                                y:train_set_y[batch_size*index:(index+1)*batch_size]})

    cost_train = function(inputs=[index],
                            outputs=[model.final_cost],
                            givens={
                                x:train_set_x[batch_size*index:(index+1)*batch_size],
                                y:train_set_y[batch_size*index:(index+1)*batch_size]})
                            
    n_train_batches = train_set_x.get_value().shape[0]/batch_size
    n_valid_batches = valid_set_x.get_value().shape[0]/batch_size
    n_test_batches = test_set_x.get_value().shape[0]/batch_size

    print(n_train_batches)
    print(n_valid_batches)
    print(n_test_batches)

    saving = Saving(['current_train_cost', 'train_cost'], n_exp)

    # best_cost = float('inf')
    for current_batch in xrange(n_exp):
        t0 = time.clock()
        mean_cost = np.asarray([cost_train(i) for i in xrange(n_train_batches)]).mean()
        [current_cost, cost_target_1, cost_target_2, cost_inverse_mapping_1, h1, h2] = one_step_train(current_batch)
        saving.add_to_channel('current_train_cost', current_batch, current_cost)
        saving.add_to_channel('train_cost', current_batch, mean_cost)
        print('batch: '+str(current_batch))
        print('mean_cost: '+str(mean_cost))
        print('train_cost: '+str(current_cost))
        print('cost target 1: '+str(cost_target_1))
        print('cost target 2: '+str(cost_target_2))
        print('cost inverse: '+str(cost_inverse_mapping_1))
        
        t1 = time.clock()
        print('time batch: '+str(t1-t0))


    if save_option:
        print('Saving results')
        filename = os.environ['TARGET_RESULTS'] + '/test_file.pkl'
        print(filename)
        file_result = open(filename, 'w')
        pickle.dump(saving.saving_dict, file_result)

    if plot_option:
        print('Ploting results')
        fig = plt.figure()
        ax = fig.add_subplot(121)
        saving.plot('current_train_cost', ax)
        ax = fig.add_subplot(122)
        saving.plot('train_cost', ax)
        plt.show()
