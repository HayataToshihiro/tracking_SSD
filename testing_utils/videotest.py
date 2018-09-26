#coding: utf-8
""" A class for testing a SSD model on a video file or webcam """

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image 
import pickle
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt
import sys
sys.path.append("..")
from ssd_utils import BBoxUtility
import math
#from karmanfilter_2d import matrix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def pro_dens_2d(x,y,mu_x,mu_y,P):
    x_c = (np.array([x,y])) - (np.array([mu_x, mu_y]))
    sigma = P[0:2,0:2]
    det = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    return np.exp(-x_c.dot(inv_sigma).dot(x_c[np.newaxis,:].T) / 2.0) / (2*np.pi*np.sqrt(det))

def in_error_ellipse(x,y,P):
    sigma_x = P[0,0]
    sigma_y = P[1,1]
    kai = 9.21034
    if((x/sigma_x)**2 + (y/sigma_y)**2 <= kai**2 ):
        return True
    else:
        return False

class Tracker:
    def __init__(self,next_ID,x,y,t):
        print " new tracker created. ID = " + str(next_ID)
        self.ID = next_ID
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.
        self.vy = 0.
        self.t = t

        self.P = np.matrix([[100,0,0,0],
                            [0,100,0,0],
                            [0,0,1000,0],
                            [0,0,0,1000]])

    def kf_motion(self,F,u):
        x = np.matrix([[self.x],[self.y],[self.vx],[self.vy]])
        x = (F * x) + u
        self.P = F * self.P * F.T

        self.x = x[0].tolist()[0][0]
        self.y = x[1].tolist()[0][0]
        self.vx = x[2].tolist()[0][0]
        self.vy = x[3].tolist()[0][0]

    def kf_measurement_update(self,measurement_x,measurement_y,H,R,I):
        x = np.matrix([[self.x],[self.y],[self.vx],[self.vy]])
        Z = np.matrix([measurement_x,measurement_y])
        error = Z.T - (H * x)
        S = H * self.P * H.T + R
        K = self.P * H.T * (S**-1)
        x = x + (K * error)
        self.P = (I - (K * H)) * self.P
        self.x = x[0].tolist()[0][0]
        self.y = x[1].tolist()[0][0]
        self.vx = x[2].tolist()[0][0]
        self.vy = x[3].tolist()[0][0]


    def update(self,x,y,t):
        self.x = x
        self.y = y
        self.t = t
                                    
    def scatter_graph(self,x,y,t):
        color = np.randam.rand(3)
        ax.scatter(x, y, t, s=5, c=color)


class VideoTest(object):
    """ Class for testing a trained SSD model on a video file and show the
        result in a window. Class is designed so that one VideoTest object 
        can be created for a model, and the same object can then be used on 
        multiple videos and webcams.      
        
        Arguments:
            class_names: A list of strings, each containing the name of a class.
                         The first name should be that of the background class
                         which is not used.
                         
            model:       An SSD model. It should already be trained for 
                         images similar to the video to test on.
                         
            input_shape: The shape that the model expects for its input, 
                         as a tuple, for example (300, 300, 3)    
                         
            bbox_util:   An instance of the BBoxUtility class in ssd_utils.py
                         The BBoxUtility needs to be instantiated with 
                         the same number of classes as the length of        
                         class_names.
    
    """
    
    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_classes)
        self.next_ID = 0
        # Create unique and somewhat visually distinguishable bright
        # colors for the different classes.
        self.class_colors = []
        for i in range(0, self.num_classes):
            # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col) 
        
    def run(self, video_path = 0, start_frame = 0, conf_thresh = 0.6):
        """ Runs the test on a video (or webcam)             
        
        # Arguments
        video_path: A file path to a video to be tested on. Can also be a number, 
                    in which case the webcam with the same number (i.e. 0) is 
                    used instead
                    
        start_frame: The number of the first frame of the video to be processed
                     by the network. 
                     
        conf_thresh: Threshold of confidence. Any boxes with lower confidence 
                     are not visualized.
                    
        """
    
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        # Compute aspect ratio of video     
        #msvidw = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        #vidh = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vidar = vidw/vidh
        
        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_frame)
            
        accum_time = 0
        video_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        
        gx, gy, gt ,gid = [], [], [], []

        #4 point designation
        #w=6.
        #h=6.
        w=4.3
        h=5.4
        
        #pts1 = np.float32([[383,158],[730,225],
        #                   [116,285],[500,436]])
        pts1 = np.float32([[650,298],[1275,312],
                           [494,830],[1460,845]])
        pts1 *= self.input_shape[1]/vidh
        pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        
        Homography = cv2.getPerspectiveTransform(pts1,pts2)

        
        dt = 1/vid.get(cv2.CAP_PROP_FPS)
        u = np.matrix([[0.],[0.],[0.],[0.]])
        F = np.matrix([[ 1, 0,dt, 0],
                       [ 0, 1, 0,dt],
                       [ 0, 0, 1, 0],
                       [ 0, 0, 0, 1]])
        H = np.matrix([[ 1, 0, 0, 0],
                       [ 0, 1, 0, 0]])
        R = np.matrix([[dt, 0],
                       [ 0,dt]])
        I = np.matrix([[ 1, 0, 0, 0],
                       [ 0, 1, 0, 0],
                       [ 0, 0, 1, 0],
                       [ 0, 0, 0, 1]])

        
        trackers = []
        


        while True:
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                break
                #return
            im_size = (self.input_shape[0], self.input_shape[1])#(300,300)
            resized = cv2.resize(orig_image, im_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Reshape to original aspect ratio for later visualization
            # The resized version is used, to visualize what kind of resolution
            # the network has to work with.
            to_draw = cv2.resize(resized, (int(self.input_shape[0]*vidar), self.input_shape[1]))
            
            # Use model to predict 
            inputs = [image.img_to_array(rgb)]
            tmp_inp = np.array(inputs)
            X = preprocess_input(tmp_inp)
            
            Y = self.model.predict(X)
            
            
            # This line creates a new TensorFlow device every time. Is there a 
            # way to avoid that?
            results = self.bbox_util.detection_out(Y)
            
            new_datas = []
            #new_datas.clear()
            if len(results) > 0 and len(results[0]) > 0:
                # Interpret output, only one frame is used 
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                #Bbox
                for i in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                    ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                    xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                    ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                    # Draw the box on top of the to_draw image
                    class_num = int(top_label_indices[i])
                    if((self.class_names[class_num]=='person') & (top_conf[i]>=0.996)):#0.996
                        cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), 
                                      self.class_colors[class_num], 2)
                        text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i]) 
                        
                        
                        text_top = (xmin, ymin-10)
                        text_bot = (xmin + 80, ymin + 5)
                        text_pos = (xmin + 5, ymin)
                        cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)
                        
                        
                        #print(text , '%.2f' % video_time , ( (xmin+xmax)/2, ymax ) )
                        cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
                        cv2.circle(to_draw, ((xmin+xmax)/2, ymax), 3, (0, 0, 255), -1)
                        
                        imagepoint = [[(xmin+xmax)/2],[ymax],[1]]
                        groundpoint = np.dot(Homography, imagepoint)
                        groundpoint = (groundpoint/groundpoint[2]).tolist()
                        groundpoint[0] = groundpoint[0][0]
                        groundpoint[1] = groundpoint[1][0]
                        groundpoint[2] = groundpoint[2][0]
                        
                        if((0<=groundpoint[0]) & (groundpoint[0]<=w) & (0<=groundpoint[1]) & (groundpoint[1]<=h)):
                            print(text , '%.2f' % video_time , ('%.2f' % groundpoint[0] , '%.2f' % groundpoint[1]) )
                            gx.append(groundpoint[0])
                            gy.append(groundpoint[1])
                            gt.append(video_time)
                            new_datas.append([gx[-1],gy[-1],gt[-1],0])


            #update
            #for i in range(len(trackers)):
            #    for j in range(len(new_datas)):
            #        trackers[i].kf_motion(F,u)
            #        distance = math.sqrt((trackers[i].x-new_datas[j][0])**2 + (trackers[i].y-new_datas[j][1])**2)
            #        if(distance<=1.0):
            #            trackers[i].update(new_datas[j][0],new_datas[j][1],video_time)
            #            gid.append(trackers[i].ID)
            #            new_datas[j][3]=1
            
            for i in range(len(trackers)):
                trackers[i].kf_motion(F,u)

            for i in range(len(trackers)):
                for j in range(len(new_datas)):
                    if(in_error_ellipse(trackers[i].x-new_datas[j][0],trackers[i].y-new_datas[j][1],trackers[i].P)):
                        trackers[i].kf_measurement_update(new_datas[j][0],new_datas[j][1],H,R,I)
                        trackers[i].update(trackers[i].x,trackers[i].y,video_time)
                        gid.append(trackers[i].ID)
                        new_datas[j][3]=1

            if(len(trackers)):
                print "(%.2f, %.2f, %.2f, %.2f)" % (trackers[0].x, trackers[0].y, trackers[0].vx, trackers[0].vy)
                print trackers[0].P



            #scores = [[0 for i in range(len(new_datas))] for j in range(len(trackers))]
            #for i in range(len(trackers)):
            #    trackers[i].kf_motion(F,u)
            #    for j in range(len(new_datas)):
            #        scores[i][j] = pro_dens_2d(new_datas[j][0],new_datas[j][1],trackers[i].x,trackers[i].y,trackers[i].P)
                    

            #generate new tracker
            for i in range(len(new_datas)):
                if(new_datas[i][3]==0):
                    newdetec = len(gx)-len(new_datas)+i
                    trackers.append(Tracker(self.next_ID,gx[newdetec],gy[newdetec],video_time))
                    gid.append(self.next_ID)
                    self.next_ID += 1
                
            
            
            
            # Calculate FPS
            # This computes FPS for everything, not just the model's execution 
            # which may or may not be what you want
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            video_time = video_time + 1/vid.get(cv2.CAP_PROP_FPS)
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            # Draw FPS in top left corner
            cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)


            cv2.line(to_draw, (pts1[0][0],pts1[0][1]),(pts1[1][0],pts1[1][1]), (100,200,100), thickness=2)
            cv2.line(to_draw, (pts1[0][0],pts1[0][1]),(pts1[2][0],pts1[2][1]), (100,200,100), thickness=2)
            cv2.line(to_draw, (pts1[3][0],pts1[3][1]),(pts1[1][0],pts1[1][1]), (100,200,100), thickness=2)
            cv2.line(to_draw, (pts1[3][0],pts1[3][1]),(pts1[2][0],pts1[2][1]), (100,200,100), thickness=2)


            cv2.imshow("SSD result", to_draw)
            cv2.waitKey(10)


        #create graph
        fig = plt.figure()
        ax=Axes3D(fig)

        
        color = np.random.rand(len(trackers),3)
        for i in range(len(gx)):
            iro = color[gid[i]]
            ax.scatter(gx[i],gy[i],gt[i],s=5,c=iro)
        #ax.scatter(gx, gy, gt, s=5, c="blue")

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('t')

        plt.show()
        
        cv2.destroyAllWindows()
        vid.release()

        return
