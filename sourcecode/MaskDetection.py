#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy
from matplotlib import pyplot
#讀取模型與參數檔
readDarknet = cv2.dnn.readNetFromDarknet("yolov3.cfg","mask.weights")
#圖層名稱、導出
layerName = readDarknet.getLayerNames()
oPutLayers = [layerName[i[0] - 1] for i in readDarknet.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("obj.names")]
#圖層顏色
colors = [(0,0,0),(135,206,235),(34,139,34)]
#模型辨識Func()
def imgDetect(frame):
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape 
    #讀取圖片(像素範圍255,1/255標準化;尺寸416*416;光罩影響不調整,bgr->rgb,縮放再裁剪)
    imgBlob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    #imgBlob當網路輸入層
    readDarknet.setInput(imgBlob)
    #圖層導出
    oPut = readDarknet.forward(oPutLayers)
    #取得辨識框
    arrClassId = []
    arrConfid = []
    arrBox = []
    for out in oPut:
        for detec in out:            
            tx, ty, tw, th, confid = detec[0:5]
            scores = detec[5:]
            classId = numpy.argmax(scores)  
            if confid > 0.3:   
                coorX = int(tx * width)
                coorY = int(ty * height)
                w = int(tw * width)
                h = int(th * height)
                #取座標
                x = int(coorX - w / 2)
                y = int(coorY - h / 2)
                arrBox.append([x, y, w, h])
                arrConfid.append(float(confid))
                arrClassId.append(classId)
                
    #畫方框
    #去除多餘方框
    idx = cv2.dnn.NMSBoxes(arrBox, arrConfid, 0.3, 0.4)
    #框住Object 
    for i in range(len(arrBox)):
        if i in idx:
            x, y, w, h = arrBox[i]
            label = str(classes[arrClassId[i]])
            color = colors[arrClassId[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y -5), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
    return img
    #cv2.FONT_HERSHEY_DUPLEX


# In[ ]:


import cv2
import imutils
import time

camera = cv2.VideoCapture(0)

while True:
    hasFrame, frame = camera.read()
    img = imgDetect(frame)
    cv2.imshow("frame", imutils.resize(img, width=850))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
camera.release()
cv2.destroyAllWindows()

