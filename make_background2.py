import numpy
import cv2


videopath = '/home/amsl/tmp/opencv/samples/data/test.mov'
cap = cv2.VideoCapture(videopath)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


count = 0
while(cap.isOpened()):
    ret, img = cap.read()
    resized = cv2.resize(img, (16*50,9*50))
    
    fgmask = fgbg.apply(resized)
    
    #ex
    #fgmask = [[0,  0,  0,  0,  0,  0,  0],
    #          [0,  0,  0,  0,  0,  0,  0],
    #          [0,  0,255,255,  0,  0,  0],
    #          [0,  0,255,255,  0,  0,  0],
    #          [0,  0,  0,  0,  0,  0,  0]]



    cv2.imshow('fgbg',fgmask)

    count += 1
    if count > 300:
        break
    if cv2.waitKey(1) % 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cv2.waitKey()
