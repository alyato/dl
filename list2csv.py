import csv
import random
def list2csv():
    with open('/data/lisa/data/icml_2013_emotions/test.csv') as testcsv:
        reader = csv.reader(testcsv)
        num=0
        reader=list(reader)
        testData=[]
        for row in reader:
            pix=[]
            getPixel = map(lambda e:int(e),row[1].split())
            xx=random.sample(getPixel,int(len(getPixel)*0.25))
            xx=' '.join(str(xl) for xl in xx)
            pix.append(row[0])
            pix.append(xx)
            pix.append(row[2])
            testData.append(pix)
        #print len(testData)
        #print testData[0]
        #print testData[1]
        #print testData[2]
        #print testData
        return testData
if __name__=='__main__':
    testData=list2csv()
    with open('test0.25.csv','wb') as f:
        writer=csv.writer(f)
        writer.writerow(testData)
