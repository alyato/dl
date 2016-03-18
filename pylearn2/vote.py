import numpy as np
from getRandomSumofcsv import getRandomTraincsv
from pp import forecastOnebyOne
def vote():
    testData=getRandomTraincsv()
    modelpath=['sm_best1.pkl','sm_best2.pkl','sm_best3.pkl','sm_best4.pkl','sm_best5.pkl','sm_best6.pkl','sm_best7.pkl','sm_best8.pkl','sm_best9.pkl','sm_best10.pkl']
    r=[]
    for row in testData:
        rr=[]
        for path in modelpath:
            res =  forecastOnebyOne(path,np.asarray(row[1:]))
            rr.append(res)
        rset=set(rr)
        rdic={}
        for row in rset:
            rdic[row]=rr.count(row)
        d=sorted(rdic.items(),lambda x,y:cmp(x[1],y[1]),reverse=True)
        r.append(d)
    print r
    #print r
    #t0=testData[0]
    #ty=t0[0]
    #tx=t0[1:]
    #tx=np.asarray(tx)
    ##tx=tx.reshape(1,len(tx))
    ##print '...........'
    #for row in modelpath:
    #    res = forecastOnebyOne(row,tx)
    #    r.append(res)
    #print r

if __name__=='__main__':
    vote()
