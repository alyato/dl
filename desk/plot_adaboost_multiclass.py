
#from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt
#import numpy as np
#from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
#from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append('/home/hey/Desktop/pic')
sys.path.append('/home/hey/Desktop/MachineLearning-master/DeepLearning Tutorials/dive_into_keras')
from get_train_valid_test_set_from_oliPkl import getPics
from cm import CnnModel
from data import load_data
from keras.utils import np_utils
import random
data,label = load_data()
label = np_utils.to_categorical(label,10)
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label=label[index]
(X_train,X_test)=(data[0:30000],data[30000:])
(y_train,y_test)=(label[0:30000],label[3000:])
#X_train, X_test = getPics().trainData,getPics().testData
#y_train, y_test = getPics().trainLabel,getPics().testLabel
#print X_train.shape
#print y_train.shape
bdt_discrete = AdaBoostClassifier(
    CnnModel(),
    n_estimators=500,
    learning_rate=0.3,
    algorithm="SAMME")
bdt_discrete.fit(X_train, y_train)
discrete_test_errors = []

for  discrete_train_predict in bdt_discrete.staged_predict(X_test):
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))

n_trees_discrete = len(bdt_discrete)
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
        'b',alpha = .5)
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

plt.subplots_adjust(wspace=0.25)
plt.show()
