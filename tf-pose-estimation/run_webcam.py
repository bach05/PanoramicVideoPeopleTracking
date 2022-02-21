import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


IM_W = 1920
IM_H = 960



FOV_RATIO = 368.0 / 432.0



def on_fov_x(val):
    global FOV_X
    FOV_X = val*1.0
    global FOV_Y
    global FOV_RATIO
    FOV_Y = FOV_X * FOV_RATIO

def on_center_x(val):
    global CROP_CENTER
    CROP_CENTER = (int(val), CROP_CENTER[1])

def on_center_y(val):
    return
    global CROP_CENTER
    CROP_CENTER = (CROP_CENTER[0], int(val))

def on_target(val):
    global TRACK_TARGET
    TRACK_TARGET = (val == 1)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=5)

    parser.add_argument('--resize', type=str, default='640x640',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    global FOV_X
    FOV_X = 60.0
    global FOV_Y
    FOV_Y = FOV_X * FOV_RATIO
    global CROP_CENTER
    CROP_CENTER = (IM_W/2, IM_H/2)
    FOV_W = FOV_X * IM_W / 360
    FOV_H = FOV_Y * IM_H / 180
    CROP_RATIO = (FOV_W / IM_W, FOV_H / IM_H)#(0.20, 0.25)
    CROP_SIZE = (int(IM_W * CROP_RATIO[0]), int(IM_H * CROP_RATIO[1]))

    global TRACK_TARGET
    TRACK_TARGET = False

    cv2.imshow('tf-pose-estimation result', image)
    image2 = image[CROP_CENTER[1]-CROP_SIZE[1]/2:CROP_CENTER[1]+CROP_SIZE[1]/2, CROP_CENTER[0]-CROP_SIZE[0]/2:CROP_CENTER[0]+CROP_SIZE[0]/2].copy()
    cv2.imshow('tf-pose-estimation result2', image2)

    cv2.createTrackbar('FOV_X', 'tf-pose-estimation result2' , 0, 360, on_fov_x)
    cv2.createTrackbar('CENTER_X', 'tf-pose-estimation result2' , 0, IM_W, on_center_x)
    cv2.createTrackbar('CENTER_Y', 'tf-pose-estimation result2' , 0, IM_H, on_center_y)
    cv2.createTrackbar('TARGET', 'tf-pose-estimation result2' , 0, 1, on_target)    


    while True:

        ret_val, image = cam.read()

        #logger.debug('image process+')
        

        if(TRACK_TARGET):

            print(FOV_X)

            FOV_W = FOV_X * IM_W / 360
            FOV_H = FOV_Y * IM_H / 180

            CROP_RATIO = (FOV_W / IM_W, FOV_H / IM_H)#(0.20, 0.25)

            CROP_SIZE = (int(IM_W * CROP_RATIO[0]), int(IM_H * CROP_RATIO[1]))

            assert( CROP_SIZE[0] <= IM_W and CROP_SIZE[1] <= IM_H )            

            CC = (CROP_CENTER[0], CROP_CENTER[1])

            from_x = CC[0]-CROP_SIZE[0]/2
            to_x = CC[0]+CROP_SIZE[0]/2

            if(from_x < 0):
                CC = (CROP_SIZE[0]/2, CC[1])

            if(to_x > IM_W):
                CC = (IM_W - CROP_SIZE[0]/2, CC[1]) 
            
            print("from_x : ", from_x)
            print("to_x : ", to_x)
            
            image2 = image[CC[1]-CROP_SIZE[1]/2:CC[1]+CROP_SIZE[1]/2, CC[0]-CROP_SIZE[0]/2:CC[0]+CROP_SIZE[0]/2].copy()

            print('original image (%4d, %4d) -> cropped image (%4d, %4d)' % (IM_W, IM_H, image2.shape[1], image2.shape[0]))

            OFF_X = CROP_CENTER[0]-CROP_SIZE[0]/2
            OFF_Y = CROP_CENTER[1]-CROP_SIZE[1]/2

            cv2.imshow('tf-pose-estimation result2', image2)

            humans = e.inference2([image, image2], OFF_X, OFF_Y, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        else:

            humans = e.inference3(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        
        
        #logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        cv2.imshow('tf-pose-estimation result', image)
        

        
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()
