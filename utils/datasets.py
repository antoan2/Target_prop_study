import theano
import theano.tensor as T
import gzip
import cPickle
import numpy as np

def load_dataset(file_name):

    data_file = open(file_name, 'rb')
    train_set, valid_set, test_set = cPickle.load(data_file)
    data_file.close()

    def split_data_label(data_xy):

        data_x, data_y = data_xy
        data_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        data_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

        return data_x, T.cast(data_y, 'int32')
    
    train_set_x, train_set_y = split_data_label(train_set)
    valid_set_x, valid_set_y = split_data_label(valid_set)
    test_set_x, test_set_y = split_data_label(test_set)

    rval = [(train_set_x, train_set_y),
            (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval
