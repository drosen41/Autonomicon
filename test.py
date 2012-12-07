import numpy
import os
import theano
import theano.tensor as T
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams
from basic_logistic import learn

from util import load_data, chunkify, shared_dataset
from utils import tile_raster_images

import PIL.Image
from autoencoder import dA, train_dA

# Here we train a maxpooling 1-layer dA, and classify using logistic regression over the dA hidden layer

cacheFile='da.pkl'
d = load_data()
train_x, train_y = d[0]
test_x, test_y = d[1]
if(os.path.isfile(cacheFile)):
    cF = open(cacheFile)
    da = cPickle.load(cF)
else:
    da=train_dA(train_x, learning_rate=0.1, training_epochs=300, batch_size=30,corruption_level=.3,rel_hidden=.6,chunk=5)
    cPickle.dump(da, open(cacheFile,'w'))


# Create output of training data using trained dA
l=train_x.get_value(borrow=True).shape[0]
output_train_x = [0]*l
for i in range(l):
    print i/float(l)
    output_train_x[i]=da.get_hidden_values_max_pooled(train_x[i],2)
output_train_x = numpy.dstack(output_train_x)
output_train_x = numpy.squeeze(output_train_x)
output_train_x = numpy.transpose(output_train_x)
shared_x = theano.shared(numpy.asarray(output_train_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
data = [shared_x, train_y[:output_train_x.shape[0]]]

# Create output of test data using trained dA
l=test_x.get_value(borrow=True).shape[0]
output_test_x = [0]*l
for i in range(l):
    print i/float(l)
    output_test_x[i]=da.get_hidden_values_max_pooled(test_x[i],2)
output_test_x = numpy.dstack(output_test_x)
output_test_x = numpy.squeeze(output_test_x)
output_test_x = numpy.transpose(output_test_x)
shared_x_t = theano.shared(numpy.asarray(output_test_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
data_test = [shared_x_t, test_y[:output_test_x.shape[0]]]

learn(data,data_test)
