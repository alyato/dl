#!/usr/bin/env python
# -*- coding: utf-8 -*-
from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano
import numpy as np
rng = numpy.random.RandomState(23455)


input = T.tensor4(name = 'input')


w_shape = (2,3,9,9) 

w_bound = numpy.sqrt(3*9*9)
W = theano.shared(numpy.asarray(rng.uniform(low = -1.0/w_bound, high = 1.0/w_bound,size = w_shape),
                                dtype = input.dtype),name = 'W')

b_shape = (2,)

b = theano.shared(numpy.asarray(rng.uniform(low = -.5, high = .5, size = b_shape),
                                dtype = input.dtype),name = 'b')
                                
conv_out = conv.conv2d(input,W)


output = T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x'))

f = theano.function([input],output)
