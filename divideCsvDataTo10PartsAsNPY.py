import random
import numpy as np
from getRandomSumofcsv import getRandomSumofcsv
def divide10Parts():
    trainData = getRandomSumofcsv()
    trainData_random=random.sample(trainData,len(trainData))
    tdr1=trainData_random[:3229]
    tdr1_x_list=[]
    tdr1_y_list=[]
    for row in tdr1:
        y,x=(row[0],row[1:])
        tdr1_x_list.append(x)
        tdr1_y_list.append([y])
    tdr1_x=np.asarray(tdr1_x_list)    
    tdr1_y=np.asarray(tdr1_y_list)
    #np.save('tdr1.X',tdr1_x)
    #np.save('tdr1.Y',tdr1_y)
    print 'hahahaha',tdr1_x.shape,tdr1_y.shape
    tdr2=trainData_random[3229:3229*2]
    tdr2_x_list=[]
    tdr2_y_list=[]
    for row in tdr2:
        y,x=(row[0],row[1:])
        tdr2_x_list.append(x)
        tdr2_y_list.append([y])
    tdr2_x=np.asarray(tdr2_x_list)    
    tdr2_y=np.asarray(tdr2_y_list)
    #np.save('tdr2.X',tdr2_x)
    #np.save('tdr2.Y',tdr2_y)
    print 'hahahaha',tdr2_x.shape,tdr2_y.shape
    tdr3=trainData_random[3229*2:3229*3]
    tdr3_x_list=[]
    tdr3_y_list=[]
    for row in tdr3:
        y,x=(row[0],row[1:])
        tdr3_x_list.append(x)
        tdr3_y_list.append([y])
    tdr3_x=np.asarray(tdr3_x_list)    
    tdr3_y=np.asarray(tdr3_y_list)
    #np.save('tdr3.X',tdr3_x)
    #np.save('tdr3.Y',tdr3_y)
    print 'hahahaha',tdr3_x.shape,tdr3_y.shape
    tdr4=trainData_random[3229*3:3229*4]
    tdr4_x_list=[]
    tdr4_y_list=[]
    for row in tdr4:
        y,x=(row[0],row[1:])
        tdr4_x_list.append(x)
        tdr4_y_list.append([y])
    tdr4_x=np.asarray(tdr4_x_list)    
    tdr4_y=np.asarray(tdr4_y_list)
    #np.save('tdr4.X',tdr4_x)
    #np.save('tdr4.Y',tdr4_y)
    print 'hahahaha',tdr4_x.shape,tdr4_y.shape
    tdr5=trainData_random[3229*4:3229*5]
    tdr5_x_list=[]
    tdr5_y_list=[]
    for row in tdr5:
        y,x=(row[0],row[1:])
        tdr5_x_list.append(x)
        tdr5_y_list.append([y])
    tdr5_x=np.asarray(tdr5_x_list)    
    tdr5_y=np.asarray(tdr5_y_list)
    #np.save('tdr5.X',tdr5_x)
    #np.save('tdr5.Y',tdr5_y)
    print 'hahahaha',tdr5_x.shape,tdr5_y.shape
    tdr6=trainData_random[3229*5:3229*6]
    tdr6_x_list=[]
    tdr6_y_list=[]
    for row in tdr6:
        y,x=(row[0],row[1:])
        tdr6_x_list.append(x)
        tdr6_y_list.append([y])
    tdr6_x=np.asarray(tdr6_x_list)    
    tdr6_y=np.asarray(tdr6_y_list)
    #np.save('tdr6.X',tdr6_x)
    #np.save('tdr6.Y',tdr6_y)
    print 'hahahaha',tdr6_x.shape,tdr6_y.shape
    tdr7=trainData_random[3229*6:3229*7]
    tdr7_x_list=[]
    tdr7_y_list=[]
    for row in tdr7:
        y,x=(row[0],row[1:])
        tdr7_x_list.append(x)
        tdr7_y_list.append([y])
    tdr7_x=np.asarray(tdr7_x_list)    
    tdr7_y=np.asarray(tdr7_y_list)
    #np.save('tdr7.X',tdr7_x)
    #np.save('tdr7.Y',tdr7_y)
    print 'hahahaha',tdr7_x.shape,tdr7_y.shape
    tdr8=trainData_random[3229*7:3229*8]
    tdr8_x_list=[]
    tdr8_y_list=[]
    for row in tdr8:
        y,x=(row[0],row[1:])
        tdr8_x_list.append(x)
        tdr8_y_list.append([y])
    tdr8_x=np.asarray(tdr8_x_list)    
    tdr8_y=np.asarray(tdr8_y_list)
    #np.save('tdr8.X',tdr8_x)
    #np.save('tdr8.Y',tdr8_y)
    print 'hahahaha',tdr8_x.shape,tdr8_y.shape
    tdr9=trainData_random[3229*8:3229*9]
    tdr9_x_list=[]
    tdr9_y_list=[]
    for row in tdr9:
        y,x=(row[0],row[1:])
        tdr9_x_list.append(x)
        tdr9_y_list.append([y])
    tdr9_x=np.asarray(tdr9_x_list)    
    tdr9_y=np.asarray(tdr9_y_list)
    #np.save('tdr9.X',tdr9_x)
    #np.save('tdr9.Y',tdr9_y)
    print 'hahahaha',tdr9_x.shape,tdr9_y.shape
    tdr10=trainData_random[3229*9:]
    tdr10_x_list=[]
    tdr10_y_list=[]
    for row in tdr10:
        y,x=(row[0],row[1:])
        tdr10_x_list.append(x)
        tdr10_y_list.append([y])
    tdr10_x=np.asarray(tdr10_x_list)    
    tdr10_y=np.asarray(tdr10_y_list)
    #np.save('tdr10.X',tdr10_x)
    #np.save('tdr10.Y',tdr10_y)
    #print len(tdr10)
    print 'hahahaha',tdr10_x.shape,tdr10_y.shape

if __name__ == '__main__':
    divide10Parts()

