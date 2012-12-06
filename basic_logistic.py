import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from util import *


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



def learn(train_set, test_set, batch_size = 50,
          learning_rate=0.13,
          n_epochs=10000):
    train_x, train_y = train_set
    test_x, test_y = test_set
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    print "Building Model"

    x = T.matrix('x')
    y = T.ivector('y')
    image_size = train_x.get_value(borrow=True).shape[1]
    classifier = LogisticRegression(input = x,
                                    n_in = image_size,
                                    n_out = 3)
    cost = classifier.negative_log_likelihood(y)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = {classifier.W: classifier.W - learning_rate * g_W,
               classifier.b: classifier.b - learning_rate * g_b}


    test_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: test_x,
                y: test_y})
    test_train_model = theano.function(inputs=[],
            outputs=classifier.errors(y),
            givens={
                x: train_x,
                y: train_y})
    index = T.lscalar()
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]}, name="train_model")

    print "Training Model"
    epoch = 0
    updates = 0
    total_cost = 0.0
    while epoch < n_epochs:
        epoch += 1
        for minibatch_idx in range(n_train_batches):
            batch_cost = train_model(minibatch_idx)
            total_cost += batch_cost
            updates += 1
            if updates % 1000 == 0:
                print "Test Errors", test_model()
                print "Train Errors", test_train_model()
if __name__ == '__main__':
    data = load_data(validation_sample = 0)
    learn(data[0], data[1])
