import numpy as np
from PIL import Image as im
import cPickle as cpl
image = im.open('olivettifaces.gif')
img_ndarray=np.asarray(image,dtype='float64')/256
ofs=np.empty(400)
for r in range(20)
    for c in range(20)
        ofs[r*20+c]=np.ndarray.flatten(img_ndarray[r*57:(r+1)*57,c*47:(c+1)*47])
ofs_label=np.empty(400)
for lb in range(40)
    ofs_label[lb*10:lb*10+10]=lb
ofs_label=ofs_label.astype(np.int)
write_file=open('/home/hey/Downloads/ofs.pkl','wb')
cpl.dump(ofs,write_file,-1)
cpl.dump(ofs_label,write_file,-1)
write_file.close()
