import numpy as np
import os
import cv2
from utils import decode_netout, compute_overlap, compute_ap
from preprocessing import BatchGenerator
from backend import Yolo_v2
import torch
import torch.nn as nn

class YOLO(object):
    def __init__(self, feature_extractor,
                       input_size, 
                       labels, 
                       max_box_per_image,
                       anchors):

        self.input_size = input_size
        
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box   = len(anchors)//2
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors  = anchors

        self.max_box_per_image = max_box_per_image

        # make the Yolo model
        self.Yolo = Yolo_v2(nb_box=self.nb_box, nb_class=self.nb_class, feature_extractor=feature_extractor)

        

        
        




