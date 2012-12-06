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

def shuffle(x,y):
    rs = map(lambda i : random.random(), x)
    x = map(lambda v : v[1] , sorted(zip(rs,x)))
    y = map(lambda v : v[1] , sorted(zip(rs,y)))
    return (x,y)

def load_data(sub_sample = 1.0,
              test_sample = 0.1,
              validation_sample=0.1):
    data = None
    cacheFile = 'dataset.pkl'
    if(os.path.isfile(cacheFile)):
        cF = open(cacheFile)
        data = cPickle.load(cF)
    else: 
        paths = [('imagecl/train/bicycle/', 0),
                 ('imagecl/train/car/', 1),
                 ('imagecl/train/motorbike/', 2)]
        train_images = []
        train_classes = []

        test_images = []
        test_classes = []

        validate_images = []
        validate_classes = []

        images = [train_images, test_images, validate_images]
        classes = [train_classes, test_classes, validate_classes]
        for p,c in paths:
            for im in os.listdir(p):
                # look at this guy at all?
                if random.random() > sub_sample:
                    next
                i = Image.open(p + im)
                # convert to grayscale
                i = ImageOps.grayscale(i)
                i = i.resize((28,28),Image.ANTIALIAS)
                # convert to numpy array and normalize to [0,1]
                i = numpy.array(i, dtype=numpy.float32)/255
                # i = i.flatten()
                r = random.random()
                if r < test_sample:
                    test_images.append(i)
                    test_classes.append(c)
                elif r < (test_sample + validation_sample):
                    validate_images.append(i)
                    validate_classes.append(c)
                else:
                    train_images.append(i)
                    train_classes.append(c)
        for idx in range(len(images)):
            (images[idx], classes[idx]) = shuffle(images[idx],classes[idx])
        for idx in range(len(images)):
            images[idx] = numpy.dstack(images[idx])
            images[idx] = numpy.squeeze(images[idx])        
            images[idx] = numpy.transpose(images[idx])   
        for idx in range(len(classes)):
            classes[idx] = numpy.array(classes[idx])        
        data = zip(images,classes)
        cPickle.dump(data, open(cacheFile,'w'))
    rval = map(shared_dataset, data)
    return rval

def load_data_chunk(sub_sample = 1.0,
              test_sample = 0.1,
              validation_sample=0.05,
              chunk_size = 5,new_size=50):
    """
    loads our dataset into shared variables
    """
    data = None
    cacheFile = 'dataset2.pkl'
    if(os.path.isfile(cacheFile)):
        cF = open(cacheFile)
        data = cPickle.load(cF)
    else:
        paths = [('imagecl/train/bicycle/', 0),
                 ('imagecl/train/car/', 1),
                 ('imagecl/train/motorbike/', 2)]
        train_images = []
        train_classes = []

        test_images = []
        test_classes = []

        validate_images = []
        validate_classes = []

        images = [train_images, test_images, validate_images]
        classes = [train_classes, test_classes, validate_classes]
        for p,c in paths:
            for im in os.listdir(p):
                # look at this guy at all?
                if random.random() > sub_sample:
                    next
                i = Image.open(p + im)
                # convert to grayscale
                i = ImageOps.grayscale(i)
                # rescale, but avoid most skew
                w = i.size[0]
                h = i.size[1]
                i = i.resize((new_size,(h*new_size)/w),Image.ANTIALIAS)
                # convert to numpy array and normalize to [0,1]
                i = numpy.array(i, dtype=numpy.float32)/255
                r = random.random()
                if r < test_sample:
                    i = i.flatten()
                    test_images.append(i)
                    test_classes.append(c)
                elif r < (test_sample + validation_sample):
                    i = i.flatten()
                    validate_images.append(i)
                    validate_classes.append(c)
                else:
                    train_images.append(i)
                    train_classes.append(c)
        for idx in range(len(images)):
            (images[idx], classes[idx]) = shuffle(images[idx],classes[idx])
        for idx in range(len(images)):
            images[idx] = numpy.dstack(images[idx])
            images[idx] = numpy.squeeze(images[idx])
            images[idx] = numpy.transpose(images[idx])
        for idx in range(len(classes)):
            classes[idx] = numpy.array(classes[idx])
        data = zip(images,classes)
        cPickle.dump(data, open(cacheFile,'w'))
    rval = map(shared_dataset, data)
    return rval
def chunkify(i, chunk_size):
    chunks = []
    for x in xrange(len(i)/chunk_size):
        # pull relevant strip and transpose for indexing fun.
        for y in xrange(len(i[0])/chunk_size):
            X=x*chunk_size
            Y=y*chunk_size
            chunks.append(i[x:x+chunk_size,y:y+chunk_size].flatten())
    return chunks
