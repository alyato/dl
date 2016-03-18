#this function just change train.csv to listoftrain
import csv
def changeCsvToList():
    with open('/data/lisa/data/icml_2013_emotions/test.csv') as testcsv:
        reader = csv.reader(testcsv)
        header = True
        listOftrain = list(reader)
        for row in listOftrain:
            if header:
                header=False
                continue
            getRowFromTest=map(lambda elem:int(elem),row[1].split())
            print len(getRowFromTest)
        #print type(listOftrain)
        #print len(listOftrain)
        #print len(listOftrain[0])
        ##print listOftrain[1]
        #a=listOftrain[0]
        #print type(a[0])
        #print len(a[0])
        #print a[0]
        #print type(a[1])
        #print len(a[1])
        #print a[1]
        #print type(a[2])
        #print len(a[2])
        #print a[2]

if __name__ == '__main__':
    changeCsvToList()

