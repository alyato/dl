#!/usr/bin/env python
#-*- coding: utf-8 -*-
from theano.tensor.nnet import conv
import theano.tensor as T
import theano,numpy
import numpy as np
from PIL import Image
import pylab
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/hey/Desktop/convfilter')
from convFunction import f


img1 = Image.open('/home/hey/Downloads/face.jpg')
width1,height1 = img1.size
img1 = numpy.asarray(img1, dtype = 'float32')/256.
img1_rgb = img1.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height1,width1)



img2 = Image.open('/home/hey/Downloads/face.jpg')
width2,height2 = img2.size
img2 = numpy.asarray(img2,dtype = 'float32')/256.
img2_rgb = img2.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height2,width2)



minibatch_img = numpy.concatenate((img1_rgb,img2_rgb),axis = 0)
filtered_img = f(minibatch_img)

imgs=filtered_img.reshape(filtered_img.shape[0]*filtered_img.shape[1],
                          filtered_img.shape[2],filtered_img.shape[3])

#def getPic():
for img in imgs:
  #  image = Image.fromarray(img*256)
    image = Image.fromarray(img)
   # print (img)
    #plt.plot(imgz)
    #imgz=imgz.swapaxes(0,1)
    pylab.axis('off')
    pylab.imshow(image)
    pylab.show()
    #plt.show()


#pylab.subplot(2,3,1);pylab.axis('off');
#pylab.imshow(img1)
#pylab.subplot(2,3,4);pylab.axis('off');
#pylab.imshow(img2)
#
#
#pylab.gray()
#
#
#pylab.subplot(2,3,2); pylab.axis("off")
#pylab.imshow(filtered_img[0,0,:,:]) #0:minibatch_index; 0:1-st filter
#pylab.subplot(2,3,3); pylab.axis("off")
#pylab.imshow(filtered_img[0,1,:,:]) #0:minibatch_index; 1:1-st filter
#
#
#
#pylab.subplot(2,3,5); pylab.axis("off")
#pylab.imshow(filtered_img[1,0,:,:]) #0:minibatch_index; 0:1-st filte
#pylab.subplot(2,3,6); pylab.axis("off")
#pylab.imshow(filtered_img[1,1,:,:]) #0:minibatch_index; 1:1-st filter
#
#
#
#pylab.show()

