from sklearn.externals.six.moves import zip
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
x,y = make_gaussian_quantiles(n_samples=13000,n_features=10,n_classes=3,random_sate=1)
n_split=3000
xtrain,xtest=x[:n_split],x[n_split:]
ytrain,ytest=y[:n_split],y[n_split:]
bdt_real=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1)
