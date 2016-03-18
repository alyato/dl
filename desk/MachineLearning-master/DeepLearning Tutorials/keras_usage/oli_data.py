#coding=utf-8
import numpy 
from PIL import Image
def load_data():
    img = Image.open('olivettifaces.gif')
    
    img_ndarray = numpy.asarray(img, dtype='float32')
    #img_ndarray = numpy.asarray(img, dtype='float64')/256
    faces=numpy.empty((400,2679))
    #faces = numpy.empty((400,1,47,57))
    for row in range(20):
       for column in range(20):
            faces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])
    faces = faces.reshape((400,1,57,47)) 
    label=numpy.empty((400,),dtype = 'uint8')
    #label=numpy.empty(400)
    for i in range(40):
           label[i*10:i*10+10]=i
    return faces,label
    #label=label.astype(numpy.int)
    #print label.shape
    #print faces.shape
    #print type(faces)
    #print type(faces[0])
    #print 'faces[0] is '
    #print len(faces[0])
    ##print type(faces[1])
    #print 'faces[1] is '
    #print len(faces[1])
    #print 'the shape of faces is '
    #print faces.shape
    #print 'the len of faces is ' 
    #print len(faces)
    #print faces
       
              #分成训练集、验证集、测试集，大小如下$
    #train_data=numpy.empty((320,2679))
    #train_label=numpy.empty(320)
    #valid_data=numpy.empty((40,2679))
    #valid_label=numpy.empty(40)
    #test_data=numpy.empty((40,2679))
    #test_label=numpy.empty(40)

    #for i in range(40):
    #    train_data[i*8:i*8+8]=faces[i*10:i*10+8]
    #    train_label[i*8:i*8+8]=label[i*10:i*10+8]
    #    valid_data[i]=faces[i*10+8]
    #    valid_label[i]=label[i*10+8]
    #    test_data[i]=faces[i*10+9]
    #    test_label[i]=label[i*10+9]

    #return ((train_data,train_label),(valid_data,valid_label),(test_data,test_label))
#if __name__=='__main__':
#    load_data()
#    data = load_data()
#    print len(data)
#    train = data[2]
#    print len(train)
#    trainD=train[0]
#    print trainD.shape

