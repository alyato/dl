import theano
import numpy as np
from theano import tensor as T
x=T.vector('x')
classes = T.scalar('n_classes')
onehot = T.eq(x.dimshuffle(0,'x'),T.arange(classes).dimshuffle('x',0))
oneHot = theano.function([x,classes],onehot)
y = T.matrix('y')
y_pred = T.matrix('y_pred')
confMat = T.dot(y.T,y_pred)
confusionMatrix = theano.function(inputs=[y,y_pred],outputs=confMat)

def confusion_matrix(x,y,n_class):
        return confusionMatrix(oneHot(x,n_class),oneHot(y,n_class))
#y_true =[2, 0, 2, 2, 0, 1]
#y_out = [0, 0, 2, 2, 0, 2]
#confusion_matrix(y_true,y_out,3)
