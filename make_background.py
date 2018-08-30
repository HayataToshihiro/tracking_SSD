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

def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (bgr[2], bgr[1], bgr[0])


def rgb_to_hsv(r, g, b):
    hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
    return (hsv[0], hsv[1], hsv[2])

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
                pixelValues[h][w] = rgb_to_hsv(img[h,w][2],img[h,w][1],img[h,w][0])
    data.append(pixelValues)
    for i in range(len(data)):    
        print data[i][0][0]
    print "---"

    #各ピクセルの最頻値計算
    for h in range(int(H)):
        for w in range(int(W)):
            
            #Blue  = []
            #Green = []
            #Red   = []
            hh = []
            ss = []
            vv = []
            for frame in range(len(data)):
                #Blue.append(data[frame][h][w][0])
                #Green.append(data[frame][h][w][1])
                #Red.append(data[frame][h][w][2])
                hh.append(data[frame][h][w][0])
                ss.append(data[frame][h][w][1])
                vv.append(data[frame][h][w][2])

            #Bcount = np.bincount(Blue)
            #Gcount = np.bincount(Green)
            #Rcount = np.bincount(Red)
            Hcount = np.bincount(hh)
            Scount = np.bincount(ss)
            Vcount = np.bincount(vv)

            #Bmode = np.argmax(Bcount)
            #Gmode = np.argmax(Gcount)
            #Rmode = np.argmax(Rcount)
            Hmode = np.argmax(Hcount)
            Smode = np.argmax(Scount)
            Vmode = np.argmax(Vcount)
            
            #background[h][w][0] = int(Bmode)
            #background[h][w][1] = int(Gmode)
            #background[h][w][2] = int(Rmode)
            background[h][w][0] = int(hsv_to_rgb(Hmode,Smode,Vmode)[2])
            background[h][w][1] = int(hsv_to_rgb(Hmode,Smode,Vmode)[1])
            background[h][w][2] = int(hsv_to_rgb(Hmode,Smode,Vmode)[0])

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

