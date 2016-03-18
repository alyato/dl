#!/usr/bin/env python
# -*- coding: utf-8 -*-
from theano.tensor.nnet import conv
import theano.tensor as T
import theano,numpy
import numpy as np
from PIL import Image
import cv2
import pylab
from multiprocessing.pool import ThreadPool


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def process_threaded(img, filters, threadn = 8):
    accum = np.zeros_like(img)
    def f(kern):
        return cv2.filter2D(img, cv2.CV_8UC3, kern)
    pool = ThreadPool(processes=threadn)
    for fimg in pool.imap_unordered(f, filters):
        np.maximum(accum, fimg, accum)
    return accum

if __name__ == '__main__':
    import sys
    sys.path.append('/home/hey/opencv-3.0.0/samples/python2')
    from common import Timer
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
    minibatch_img = numpy.concatenate((img1_rgb,img2_rgb,),axis = 0)
    filtered_img = f(minibatch_img)
    imgs = filtered_img.reshape(filtered_img.shape[0]*filtered_img.shape[1],
                                filtered_img.shape[2],filtered_img.shape[3])
   # print ('imgs 的维度')
   # print (imgs.shape)
    print __doc__
    try:
        img = sys.argv[1]
    except:
      for img in imgs:
        #  img_fn = Image.fromarray(imgx*256)
        # if img_fn is None:
        #     print 'Failed to load image file:',img_fn
        #    sys.exit(1)
          filters = build_filters()
          with Timer('running single-threaded'):
              res1 = process(img,filters)
          with Timer('running multi-threaded'):
              res2 = process_threaded(img,filters)
          print 'res1==res2:',(res1==res2).all
          cv2.imshow('img',img)
          cv2.imshow('result',res2)
          cv2.waitKey()
          cv2.destroyAllWindows()
     # img_fn=image
     # pylab.axis('off')
     # pylab.imshow(image)
     # pylab.show()
        #img_fn = '../Downloads/face.jpg'
   #img = cv2.imread(img_fn)
  # if img is None:
  #     print 'Failed to load image file:', img_fn
  #     sys.exit(1)
  # filters = build_filters()

  # with Timer('running single-threaded'):
  #     res1 = process(img, filters)
  # with Timer('running multi-threaded'):
  #     res2 = process_threaded(img, filters)

  # print 'res1 == res2: ', (res1 == res2).all()
  # cv2.imshow('img', img)
  # cv2.imshow('result', res2)
  # cv2.waitKey()
  # cv2.destroyAllWindows()
