a=[1,1,2,2,3,3,3,3,4,4,4]
b=set(a)
c={}
for row in b:
    #c[a.count(row)]=row
    c[row]=a.count(row)
print c
e=c.items()
print e
d=sorted(c.items(),lambda x,y: cmp(x[1],y[1]),reverse=True)
print d[0][0]
