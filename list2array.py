import numpy as np
import random
a=[[0,1,1,1,1,1,1],
        [1,2,2,2,2,2,2],
        [2,3,3,3,3,3,3],
        [3,4,4,4,4,4,4],
        [4,5,5,5,5,5,5],
        [5,6,6,6,6,6,6],
        [6,7,7,7,7,7,7],
        [7,8,8,8,8,8,8],
        [8,9,9,9,9,9,9],
        [9,11,11,11,11,11]]
print len(a)
b=random.sample(a,len(a))
b1=b[:4]
print b1
b2=b[4:8]
print b2
b3=b[8:]
print b3
b1_y_list=[]
b1_x_list=[]
b2_y_list=[]
b2_x_list=[]
b3_y_list=[]
b3_x_list=[]
for row in b1:
    y,x=(row[0],row[1:])
    b1_y_list.append([y])
    b1_x_list.append(x)
b1_x=np.asarray(b1_x_list)
b1_y=np.asarray(b1_y_list)
#np.save('b1_x',b1_x)
print b1_y
for row in b2:
    y,x=(row[0],row[1:])
    b2_y_list.append([y])
    b2_x_list.append(x)
b2_x=np.asarray(b2_x_list)
b2_y=np.asarray(b2_y_list)
print b2_y
for row in b3:
    y,x=(row[0],row[1:])
    b3_y_list.append([y])
    b3_x_list.append(x)
b3_x=np.asarray(b3_x_list)
b3_y=np.asarray(b3_y_list)
print b3_y
print b3_y.shape
