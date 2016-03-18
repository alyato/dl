
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano
import numpy as np
import pylab
from PIL import Image
from convFunction import f

#-------------img1---------------
img1 = Image.open('/home/hey/Downloads/face.jpg')
width1,height1 = img1.size
img1 = numpy.asarray(img1, dtype = 'float32')/256. 


img1_rgb = img1.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height1,width1) 


#-------------img2---------------
img2 = Image.open('/home/hey/Downloads/face.jpg')
width2,height2 = img2.size
img2 = numpy.asarray(img2,dtype = 'float32')/256.
img2_rgb = img2.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height2,width2) 

#------------img3----------------
img3 = Image.open('/home/hey/Downloads/face.jpg')
width3,height3 = img3.size
img3 = numpy.asarray(img3, dtype = 'float32')/256. 
img3_rgb = img3.swapaxes(0,2).swapaxes(1,2).reshape(1,3,height1,width1) 


minibatch_img = numpy.concatenate((img1_rgb,img2_rgb,img3_rgb),axis = 0)
filtered_img = f(minibatch_img)
#filtered_img = f(filtered_img)




# plot original image and two convoluted results
pylab.subplot(3,3,1);pylab.axis('off');
pylab.imshow(img1)

pylab.subplot(3,3,4);pylab.axis('off');
pylab.imshow(img2)

pylab.subplot(3,3,7);pylab.axis('off');
pylab.imshow(img3)


pylab.gray()

pylab.subplot(3,3,2); pylab.axis("off")
pylab.imshow(filtered_img[0,0,:,:]) 

pylab.subplot(3,3,3); pylab.axis("off")
pylab.imshow(filtered_img[0,1,:,:]) 

pylab.subplot(3,3,5); pylab.axis("off")
pylab.imshow(filtered_img[1,0,:,:]) 

pylab.subplot(3,3,6); pylab.axis("off")
pylab.imshow(filtered_img[1,1,:,:])

pylab.subplot(3,3,8); pylab.axis("off")
pylab.imshow(filtered_img[2,0,:,:])

pylab.subplot(3,3,9); pylab.axis("off")
pylab.imshow(filtered_img[2,1,:,:])
#pylab.subplot(2,3,5); pylab.axis("off")
#pylab.imshow(filtered_img[0,1,:,:])
pylab.show()
