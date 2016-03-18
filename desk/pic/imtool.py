# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
def get_imlist(path):
#返回目录中所有jpg格式的文件列表'
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]
    '''
    for f in os.path.listdir(path)
        if f.endswith('.jpg')
            return os.path.join(path,f)
    '''
def imresize(im,size):
    """使用PIL对象重新定义图像数组的大小"""
    img=Image.fromarray(uint8(im))
    return np.array(img.resize(size))
def histeq(img ,nbr_bins=256):
    """对一副图像进行直方图均衡化"""
    imhist,bins=np.histogram(img.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum()
    cdf = 255*cdf/cdf[-1]
    img1 = np.interp(img.flatten(),bins[:-1],cdf)
    return img1.reshape(img.shape),cdf 
#if  __name__ == "__main__":
#    a=[]
#    print type(a)
#    print a
#    a=get_imlist('/home/hey/Downloads/face')
#    #return a
#    print type(a)
#    print a
