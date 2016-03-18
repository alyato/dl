#coding:utf-8
import os 
from PIL import Image im
import numpy as np
def loadData():
    data=np.empty(
            (42000,1,28,28),
            dtype='float32'
            )
    label=np.empty(
            (42000,),
            dtype='float32'
            )
    imgs=os.listdir('/home/hey/Downloads/mnist')
    num=len(imgs)
    for i in range(num):
        img=im.open('/home/hey/Downloads/mnist'+imgs[i])
        arr=np.asarray(img,dtype='float32')
        data[i,:,:,:]=arr
        label[i]=int(imgs[i].split('.')[0])
    return data,label

