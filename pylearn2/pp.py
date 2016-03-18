#!/usr/bin/env python
# encoding: utf-8
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import GlobalContrastNormalization
import Image
import numpy as np
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
def preprocessImage(path):
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((48,48),Image.ANTIALIAS)
    return np.array(img.getdata())
def forecast_by_imgpath(model_path, imgpath='', proptype='fprop'):
    return forecastOnebyOne(model_path, img=preprocessImage(imgpath), proptype=proptype)
def forecastOnebyOne(model_path,img,proptype='fprop'):
    model = serial.load(model_path)
    model.set_batch_size(1)
    img_arr=img
    #imgg=np.asarray(img_arr)
    #imgg=imgg.reshape(imgg.shape[0],1)
    #print imgg.shape
    #print imgg
    #img_arr =np.reshape(img, 2304)
    img_list=np.array([img_arr])
    #view_converter= DefaultViewConverter(shape=[48,48,1],axes=('b',0,1,'c'))
    #dense_design_martix = DenseDesignMatrix(X=img_list,y=None,view_converter=view_converter)
    #preprocessor = GlobalContrastNormalization(sqrt_bias=10,use_std=1)
    #preprocessor.apply(dense_design_martix, False)
    #img_list= dense_design_martix.get_data()
    #img_list=dense_design_martix.get_topological_view(img_list)
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    y = T.argmax(Y, axis=1)
    f = function([X], y)
    #result = f(imgg.astype(np.float32))
    result = f(img_list.astype(np.float32))
    return result[0]
def tt():
    model_path = '/home/hey/pylearn2/pylearn2/scripts/icml_2013_wrepl/emotions/softmax_best.pkl'
    forecast_by_imgpath(model_path, '/home/hey/Downloads/face.jpg')
if __name__=='__main__':
    tt()
