#coding: utf-8

import cv2
import numpy as np
from PIL import Image
import statistics
import math


Videopath = '/home/amsl/tmp/opencv/samples/data/vtest.avi'
cap = cv2.VideoCapture(Videopath)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

background = [[[0 for i in range(3)] for j in range(int(W))] for k in range(int(H))]
data = []
c=0

while(cap.isOpened()):
    ret, img = cap.read()
    #フレームデータ読み込み
    pixelValues = [[[0 for i in range(3)] for j in range(int(W))] for k in range(int(H))] 
    for h in range(int(H)):
        for w in range(int(W)):
            if not img[h,w] is None:
                pixelValues[h][w] = img[h,w]
    data.append(pixelValues)
    for i in range(len(data)):
        print data[i][0][0]
    print "---"

    #各ピクセルの最頻値計算
    for h in range(int(H)):
        for w in range(int(W)):
            
            Blue  = []
            Green = []
            Red   = []
            for frame in range(len(data)):
                Blue.append(data[frame][h][w][0])
                Green.append(data[frame][h][w][1])
                Red.append(data[frame][h][w][2])

            Bcount = np.bincount(Blue)
            Gcount = np.bincount(Green)
            Rcount = np.bincount(Red)

            Bmode = np.argmax(Bcount)
            Gmode = np.argmax(Gcount)
            Rmode = np.argmax(Rcount)

            #Bmode =statistics.mode(Blue)
            #Gmode =statistics.mode(Green)
            #Rmode =statistics.mode(Red)
            
            background[h][w][0] = int(Bmode)
            background[h][w][1] = int(Gmode)
            background[h][w][2] = int(Rmode)
    #背景生成
    background_img = Image.new('RGB',(int(W),int(H)))
    for h in range(int(H)):
        for w in range(int(W)):
            background_img.putpixel((w,h),(background[h][w][2],background[h][w][1],background[h][w][0]))
    
    
    
    
    if c==cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break
    c+=1

    #表示
    cv2.destroyAllWindows()
    #cv2.imshow('video',img)
    background_img.show()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()



#for h in range(int(H)):
#    for w in range(int(W)):
#        Blue  = []
#        Green = []
#        Red   = []
#        for frame in range(len(data)):
#            Blue.append(data[frame][h][w][0])
#            Green.append(data[frame][h][w][1])
#            Red.append(data[frame][h][w][2])
#
#        Bcount = np.bincount(Blue)
#        Gcount = np.bincount(Green)
#        Rcount = np.bincount(Red)
#
#        Bmode = np.argmax(Bcount)
#        Gmode = np.argmax(Gcount)
#        Rmode = np.argmax(Rcount)
#
#        #background[h][w][0] = Bmode
#        #background[h][w][1] = Gmode
#        #background[h][w][2] = Rmode
#        
#        background[h][w][0] = (int)np.average(Blue)
#        background[h][w][1] = (int)np.average(Green)
#        background[h][w][2] = (int)np.average(Red)
#    print background[h][w]

background_img = Image.new('RGB',(int(W),int(H)))
for h in range(int(H)):
    for w in range(int(W)):
        background_img.putpixel((w,h),(background[h][w][2],background[h][w][1],background[h][w][0]))

background_img.show()
