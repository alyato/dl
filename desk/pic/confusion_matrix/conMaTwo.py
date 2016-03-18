from conMa import confusion_matrix
y_true =[2, 0, 2, 2, 0, 1]
y_out = [0, 0, 2, 2, 0, 2]
print confusion_matrix(y_true,y_out,3)
