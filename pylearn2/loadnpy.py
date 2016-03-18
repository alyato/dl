import numpy as np
path='test.X.npy'
#a=np.load((path[:-4]+'.X.npy'))
#print a
#path ='../../../../../b1_x.npy' 
a=np.load(path)
print type(a) 
print a.shape

