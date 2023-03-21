import cv2
import numpy as np

cap = cv2.VideoCapture(0)
height=480
width=640
cap.set(3, width)
cap.set(4, height)
cap.set(10,150)

def PreProcessing(img):
    kernel=np.ones((5,5))
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(img,200,200)
    imgDialation=cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres=cv2.erode(imgDialation,kernel,iterations=1)

    return imgThres

def getContours(img):
    biggest=np.array([])
    maxarea=0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area>maxarea and len(approx)==4:
                biggest=approx
                maxarea=area
    cv2.drawContours(imgResult, biggest , -1, (255, 0, 0), 20)
    return biggest

def reorder(myPoints):
    myPoints=myPoints.reshape(4,2)
    myPointsnew=np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1)
    #print("add",add)

    myPointsnew[0]=myPoints[np.argmin(add)]
    myPointsnew[3]=myPoints[np.argmin(add)]
    diff=np.diff(myPoints, axis=1)
    myPointsnew[1]=myPoints[np.argmin(diff)]
    myPointsnew[2]=myPoints[np.argmin(diff)]

def getWarp(img,biggest):
    biggest=reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img,matrix, (width, height))

    return imgOutput

while True:
    success, img = cap.read()
    img=cv2.resize(img,(width,height))
    imgResult=img.copy()
    #imgWarp=img.copy()
    imgThres=PreProcessing(img)
    biggest=getContours(imgThres)
    if biggest.size!=0:    
        imgWarp=getWarp(img,biggest)
        cv2.imshow("Result", imgWarp)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break