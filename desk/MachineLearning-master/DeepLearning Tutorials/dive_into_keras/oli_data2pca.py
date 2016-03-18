from PIL import Image
import sys
sys.path.append('/home/hey/Desktop/pic')
from pca import pca_mod
import numpy as np
def load_data_pca():
    img = Image.open('olivettifaces.gif')
    img_array = np.asarray(img,dtype='float32')
    faces = np.empty((400,2679))
    for row in range(20):
        for col in range(20):
            faces[row*20+col] = np.ndarray.flatten(img_array[row*57:(row+1)*57,col*47:(col+1)*47])
    lowDDataMat,reconMat = pca_mod(faces)
    print 'lowDDataMat ..............'
    print lowDDataMat.shape
    print lowDDataMat.size
    print 'reconMat ..................'
    print reconMat.shape
    print reconMat.size
if __name__ == '__main__':
    load_data_pca()
