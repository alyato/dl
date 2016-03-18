import cv2
import os
def extractFace(img):
    print os.path.relpath(__file__)
    cascade=cv2.CascadeClassifier(os.path.split(os.path.realpath(__file__)[0])+'/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects=detect(gray,cascade)
    vis=img.copy()
    #draw_rects(vis,rects,(0,255,0))
def draw_rects(img,rects,color):
    for x1,y1,x2,y2 in rects:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
def detect(img,cascade):
    rects=cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=4,minSize=(40,40),flags=cv2.CASCADE_SCALE_IMAGE)
    print type(rects)
    if(len(rects)==0):
        print 'rects==0'
        return []
    rects[:,2:]+=rects[:,:2]
    return rects
#path='/home/hey/Downloads/face.jpg'
#img= cv2.imread(path)
if __name__=='__main__':
    img= cv2.imread('face.jpg')
    extractFace(img)

