import numpy
import os
import theano
import theano.tensor as T
import cPickle
from theano.tensor.shared_randomstreams import RandomStreams

from util import load_data, chunkify
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
chunk=theano.shared(5)
i = T.scalar('i',dtype='int32')
j = T.scalar('j',dtype='int32')
x = T.matrix('x')
f = theano.function([i,j],x[i:i+chunk,j:j+chunk],givens={x:train_x[0]})
print f(0,0)
print f(1,0)

out = da.get_hidden_values_max_pooled(train_x[0])

