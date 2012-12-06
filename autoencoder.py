import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from util import load_data, chunkify
from utils import tile_raster_images

import PIL.Image

class dA(object):
    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None, chunks=5):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    # this totally doesn't work atm
    def get_hidden_values_max_pooled(self, input,step=1):
        i = input.get_value(borrow=True)
        xvals = range(0,i.shape(0),step)
        yvals = range(0,i.shape(1),step)
        outputs = [[0 for y in range(len(yvals))] for x in range(len(xvals))]
        for x in range(len(xvals)):
            for y in range(0,i.shape(0),step):
                i[xvals[x]:xvals[x]+self.chunk,yvals[y]:yvals[y]+self.chunk]
                outputs[x,y]=self.get_hidden_values(shared_i).eval()
        # reshape into something maxpool will like
        shared_x = theano.shared(numpy.asarray(outputs,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
        # actually our dimensionality is off :/
        return 
                
        
    def get_reconstructed_input(self, hidden):
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam

        return (cost, updates)

def train_dA(learning_rate=0.1, training_epochs=500, batch_size=30):
    d = load_data()
    train_x, train_y = d[0]
    # transform training data into  what we want
    xs=train_x.get_value(borrow=True)
    real_train = []
    for x in xs:        
        real_train += chunkify(x,self.chunks)
    train_x = theano.shared(numpy.asarray(real_train,
                                           dtype=theano.config.floatX), borrow=True)
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images    

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    image_size = train_x.get_value(borrow=True).shape[1]
    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=image_size, n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.3,
                                        learning_rate=learning_rate)

    train_da = theano.function([index], cost, updates=updates,
         givens={x: train_x[index * batch_size:(index + 1) * batch_size]})

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)        

    image = PIL.Image.fromarray(tile_raster_images(
        X=da.W.get_value(borrow=True).T,
        img_shape=(50, 50), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')


if __name__ == '__main__':
    train_dA()

