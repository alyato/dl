import csv
def getTestPre():
    with open('sb.csv','rb') as sbcsv:
    #with open('pylearn2/pylearn2/scripts/icml_2013_wrepl/emotions/sb.csv','rb') as sbcsv:
    #with open('/home/hey/pylearn2/pylearn2/scripts/icml_2013_wrepl/emotions/sb.csv','rb') as sbcsv:
        lines= csv.reader(sbcsv)
        testPre=[]
        for line in lines:
            testPre.append(int(line[0]))
    return testPre
if __name__=='__main__':
    a=getTestPre()
    print len(a)


