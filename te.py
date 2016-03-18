import random
def getRandomSample():
    a=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7],[8,8,8],[9,9,9],[10,10,10],[11,11,11],[12,12,12],[13,13,13,13],[14,14,14],[15,15,15],[16,16,16],[17,17,17],[18,18,18],[19,19,19]]
    c=[]
    b=random.sample(a,len(a))
    for i in range(int(round(len(a)/4.0))):
        if i==4:
            c.insert(i,b[16:])
        else:
            c.insert(i,b[4*i:4*(i+1)])
    print len(c)
    print c

    #for i in range(10):
    #    #b=random.sample(a,int(len(a)*0.5))
    #    b=random.sample(a,len(a))
    #    
    #    print b
    #    print '****************************'
if __name__ == '__main__':
    getRandomSample()

