#coding=utf-8
import numpy 
from PIL import Image
import cv2
def load_data():
    img = Image.open('olivettifaces.gif')
    
    img_ndarray = numpy.asarray(img, dtype='float32')
    faces=numpy.empty((400,2679))
    for row in range(20):
       for column in range(20):
            faces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])
    faces = faces.reshape((400,57,47)) 
    i = 0
    label = None
    for pic in faces:
#        i = i + 1
        label = int(i / 10)
        index = (i % 10)
        i = i + 1
 #       Image.fromarray(pic).save('/home/hey/Downloads/olipics/(%i).(%i).jpg' %(label ,index ))
        cv2.imwrite('/home/hey/Downloads/olipics/{0}.{1}.jpg'.format(label,index),pic)
#    i = i + 1
if __name__ == '__main__':
    load_data()

    

