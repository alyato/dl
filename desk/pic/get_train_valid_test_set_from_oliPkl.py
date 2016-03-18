import cPickle as cpl
import numpy as np
class getPics():
    def __init__(self):
        self.readFile =open ('../../../pic/olivettifaces.pkl')
        #self.readFile =open ('pic/olivettifaces.pkl')
        #self.readFile =open (os.path.join(os.path.dirname(__file__), 'olivettifaces.pkl'))
        self.faces = cpl.load(self.readFile)
        self.label = cpl.load(self.readFile)
        self.readFile.close()
        self.trainData = np.empty((320,2679))
        self.trainLabel = np.empty(320)
        self.validData = np.empty((40,2679))
        self.validLabel = np.empty(40)
        self.testData = np.empty((40,2679))
        self.testLabel = np.empty(40)
        #data=[]
        for i in range (40):
            self.trainData[i*8:i*8+8]=self.faces[i*10:i*10+8]
            self.trainLabel[i*8:i*8+8]=self.label[i*10:i*10+8]
            self.validData[i]=self.faces[i*10+8]
            self.validLabel[i]=self.label[i*10+8]
            self.testData[i]=self.faces[i*10+9]
            self.testLabel[i]=self.label[i*10+9]
        
        

#if __name__=='__main__':
#    pic =getPics()
#    print pic.trainData.shape
#    train_Data=pic.trainData
#    trainData=train_Data.reshape(train_Data.shape[0],1,train_Data.shape[1]/47,train_Data.shape[1]/57)
#    print trainData.shape
