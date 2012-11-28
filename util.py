import os
from PIL import Image, ImageOps
import cPickle
import theano
import theano.tensor as T
import numpy
import random

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

def load_data(sample = 1.0):
    """
    loads our dataset into shared variables
    """
    data = None
    cacheFile = 'dataset.pkl'
    if(os.path.isfile(cacheFile)):
        cF = open(cacheFile)
        data = cPickle.load(cF)
    else: 
        paths = [('imagecl/train/bicycle/', 0),
                 ('imagecl/train/car/', 1),
                 ('imagecl/train/motorbike/', 2)]
        images = []
        classes = []
        for p,c in paths:
            for im in os.listdir(p):
                if random.random() > sample:
                    next
                i = Image.open(p + im)
                # convert to grayscale
                i = ImageOps.grayscale(i)
                i = i.resize((50,50),Image.ANTIALIAS)
                # convert to numpy array and normalize to [0,1]
                i = numpy.array(i, dtype=numpy.float32)/255
                i = i.flatten()
                images.append(i)
                classes.append(c)
        classes = numpy.array(classes)
        images = numpy.dstack(images)
        images = numpy.squeeze(images)
        data = (images,classes)
        cPickle.dump(data, open(cacheFile,'w'))
    xs, ys = shared_dataset(data)
    rval = (xs,ys)
    return rval

