import os
from PIL import Image, ImageOps
import cPickle
import numpy

def load_data():
    """
    loads our dataset into shared variables
    """

    cacheFile = 'dataset.pkl'
    if(os.path.isfile(cacheFile)):
        cF = open(cacheFile)
        return cPickle.load(cF)

    paths = [('imagecl/train/bicycle/',0),
             ('imagecl/train/car/',1),
             ('imagecl/train/motorbike/',2)]
    images = []
    classes = []
    for p,c in paths:
        for im in os.listdir(p):
            i = Image.open(p + im)
            # convert to grayscale
            i = ImageOps.grayscale(i)
            # convert to numpy array and normalize to [0,1]
            i = numpy.array(i, dtype=numpy.float32)/255
            images.append(i)
            classes.append(c)
    classes = numpy.array(classes)
    cPickle.dump(classes, open(cacheFile,'w'))
    return (images,classes)
