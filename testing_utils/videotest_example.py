import keras
import pickle
from videotest import VideoTest
import cv2
import numpy as np
import sys
sys.path.append("..")
from ssd import SSD300

#videopath = '/home/amsl/tmp/opencv/samples/data/vtest.avi' 
#videopath = '/home/amsl/tmp/opencv/samples/data/video(sample).avi'  
videopath = '/home/amsl/tmp/opencv/samples/data/test.mov' 
video = cv2.VideoCapture(videopath)
W = video.get(cv2.CAP_PROP_FRAME_WIDTH)
H = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

input_shape = (int(H),int(H),3)#(300,300,3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
NUM_CLASSES = len(class_names)

model = SSD300(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('../weights_SSD300.hdf5')
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
vid_test.run(videopath)
#vid_test.run()
