#coding:utf-8

#导入各种用到的模块组件
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD
#from keras.utils import np_utils, generic_utils
#from six.moves import range


#nb_epoch = 5
#batch_size = 100
#nb_class = 10
#加载数据
#data, label = load_data()

#label为0~9共10个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
#label = np_utils.to_categorical(label, nb_class)

#14layer, one "add" represent one layer
class CnnModel():
    def __init__(self):
            self.model = Sequential()
            self.model.add(Convolution2D(4, 1, 5, 5, border_mode='valid')) 
            self.model.add(Activation('relu'))

            self.model.add(Convolution2D(8,4, 3, 3, border_mode='valid'))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(poolsize=(2, 2)))

            self.model.add(Convolution2D(16, 8, 3, 3, border_mode='valid')) 
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(poolsize=(2, 2)))

            self.model.add(Flatten())
            self.model.add(Dense(16*4*4, 128, init='normal'))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(0.2))

            self.model.add(Dense(128, 10, init='normal'))
            self.model.add(Activation('softmax'))
            #return self.model


#############
#开始训练模型
##############
#model = create_model()
#sgd = SGD(l2=0.0,lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
#index = [i for i in range(len(data))]
#random.shuffle(index)
#data = data[index]
#label = label[index]
#(X_train,X_val) = (data[0:30000],data[30000:])
#(Y_train,Y_val) = (label[0:30000],label[30000:])
#best_accuracy = 0.0
#from data import load_data
#for e in xrange(nb_epoch):
#    #shuffle the data each epoch
#    num = len(Y_train)
#    index = [i for i in range(num)]
#    random.shuffle(index)
#    X_train = X_train[index]
#    Y_train = Y_train[index]
#
#    print('Epoch', e)
#    print("Training...")
#    batch_num = len(Y_train)/batch_size
#    progbar = generic_utils.Progbar(X_train.shape[0])
#    for i in xrange(batch_num):
#        loss,accuracy = model.train_on_batch(X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size],accuracy=True)
#        progbar.add(batch_size, values=[("train loss", loss),("train accuracy:", accuracy)] )
#
#    #save the model of best val-accuracy
#    print("Validation...")
#    val_loss,val_accuracy = model.evaluate(X_val, Y_val, batch_size=1,show_accuracy=True)
#    if best_accuracy<val_accuracy:
#        best_accuracy = val_accuracy
#save the model to the format of pkl
#cPickle.dump(model,open("/home/hey/Downloads/model.pkl","wb"))

#save the model to the format of yaml
#yamlString=model.to_yaml()
#open('/home/hey/Downloads/yamlStringModel.yaml','wb').write(yamlString)

#save the model to the format of json
#jsonString=model.to_json()
#open('/home/hey/Downloads/jsonStringModel.json','wb').write(jsonString)


#model=mfy(yamlString)
#print model
