import sys,os
sys.path.append('/home/hey/Desktop/pic')
from get_train_valid_test_set_from_oliPkl import getPics
data = getPics()
trainData,trainLabel=data.trainData,data.trainLabel
validData,validLabel=data.validData,data.validLabel
testData,testLabel = data.testData,data.testLabel
print trainData.shape

