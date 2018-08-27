#coding: utf-8

import cv2
import numpy as np
from PIL import Image

Videopath = '/home/amsl/tmp/opencv/samples/data/vtest.avi'
cap = cv2.VideoCapture(Videopath)
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

pixelValues = [[[0 for i in range(3)] for j in range(int(W))] for k in range(int(H))] 
data = []
c=0

while(cap.isOpened()):
    ret, img = cap.read()
    for h in range(int(H)):
        for w in range(int(W)):
            pixelValues[h][w] = img[h,w]
    print img[0,0]
    c+=1
    if c==cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

    data.append(pixelValues)

background = [[[0 for i in range(3)] for j in range(int(W))] for k in range(int(H))]

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

        background[h][w][0] = Bmode
        background[h][w][1] = Gmode
        background[h][w][2] = Rmode

background_img = Image.new('RGB',(int(W),int(H)))
for h in range(int(H)):
    for w in range(int(W)):
        background_img.putpixel((w,h),(background[h][w][2],background[h][w][1],background[h][w][0]))

background_img.show()
