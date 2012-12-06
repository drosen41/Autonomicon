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

cacheFile='da.pkl'
d = load_data()
train_x, train_y = d[0]
test_x, test_y = d[0]
if(os.path.isfile(cacheFile)):
    cF = open(cacheFile)
    da = cPickle.load(cF)
else:
    da=train_dA(train_x, learning_rate=0.1, training_epochs=50, batch_size=30)
    cPickle.dump(da, open(cacheFile,'w'))
output_train_x = []
l=train_x.get_value(borrow=True).shape[0]
for i in range(l/10):
    print i/float(l)
    output_train_x.append(da.get_hidden_values_max_pooled(train_x[i]))
output_train_x = numpy.dstack(output_train_x)
output_train_x = numpy.squeeze(output_train_x)
output_train_x = numpy.transpose(output_train_x)
shared_x = theano.shared(numpy.asarray(output_train_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
data = [shared_x, train_y[:output_train_x.shape[0]]]

learn(data,data)
