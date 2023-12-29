import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from model import yolo_eval, yolo4_body
from utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import cv2

from decode_np import Decode




def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline(5_000_000)
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def detect_object(image_path):
    model_path = 'sandesh_weights.h5'
    anchors_path = 'anchors_trash.txt'
    classes_path = 'class_sandesh.txt'
    img=image_path

    class_names = get_class(classes_path)

    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    model_image_size = (416, 416)

        
    conf_thresh = 0.5
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    
    image = cv2.imread(img)
    image, boxes, scores, classes = _decode.detect_image(image, True)
    # print(classes)
    dict1={}
    # for i in range(len(classes)):
    #     dict1[class_names[classes]]
    dict1["class"]=classes
    dict1["scores"]=scores
    dict1["boxes"]=boxes
    print(class_names)
    print(dict1)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
detect_object('s5.jpeg')
     
# def detect_video(self,video_path):
#     model_path = weights
#     anchors_path = 'anchors.txt'
#     classes_path = 'classes.txt'
#     video=video_path

# while True:
#     img = input('Input image filename:')
#     try:
#         image = cv2.imread(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         image, boxes, scores, classes = _decode.detect_image(image, True)
#         print(classes)
#         return image,classes
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# yolo4_model.close_session()
