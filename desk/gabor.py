import cv2
import numpy as np
import pandas as pd
import pylab as pl

df = pd.read_csv('test.csv',header=0)

df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
X = np.vstack (df['Image'].values) 

X = X.astype(np.uint8)
X = X.reshape(-1,96,96)


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

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

filters = []

for k in xrange(5):
 img = X[k]

 X[k, :, :] = image_histogram_equalization(X[k, :,:])[0]

 filters = build_filters()
 filters = np.asarray(filters)
 print filters.shape
 res1 = process(img, filters)
 cv2.imshow('result', res1)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

f = np.asarray(filters)
print 'Filters', f.shape