from getRandomSumofcsv import getRandomSumofcsv
import numpy as np
def getTest():
    testData=getRandomSumofcsv()
    #print type(testData)
    #print len(testData)
    #print type(testData[0])
    #print len(testData[0])
    tdy=[]
    tdx=[]
    for row in testData:
        y,x=(row[0],row[1:])
        tdy.append([y])
        tdx.append(x)
    tdxx=np.asarray(tdx)
    tdyy=np.asarray(tdy)
    print tdxx.shape
    print tdyy.shape
    
    np.save('test.X',tdxx)
    np.save('test.Y',tdyy)
    
if __name__=='__main__':
    getTest()
