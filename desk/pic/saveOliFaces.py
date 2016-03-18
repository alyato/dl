import numpy as np
from PIL import Image as im
import cPickle as cpl
img=im.open('/home/hey/Downloads/FaceRecognition_CNN(olivettifaces)/olivettifaces.gif')
img_ndarray=np.asarray(img,dtype='float64')/256
olivettifaces=np.empty((400,2679))
for row in range(20):
    for column in range(20):
        olivettifaces[row*20+column]=np.ndarray.flatten(img_ndarray[row*57:(row+1)*57,column*47:(column+1)*47])
olivettifaces_label=np.empty(400)
for label in range(40):
    olivettifaces_label[label*10:label*10+10]=label
olivettifaces_label=olivettifaces_label.astype(np.int)
write_file=open('/home/hey/Desktop/pic/olivettifaces.pkl','wb')
cpl.dump(olivettifaces,write_file,-1)
cpl.dump(olivettifaces_label,write_file,-1)
write_file.close()

