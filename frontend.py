import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import matplotlib.pyplot as plt
#from utils import decode_netout, compute_overlap, compute_ap
from backend import Yolo_v2
import torch
import torch.nn as nn
from dataloader import data_generator

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
        self.grid_h, self.grid_w = 13, 13
        self.max_box_per_image = max_box_per_image

        # make the Yolo model
        self.Yolo = Yolo_v2(nb_box=self.nb_box, nb_class=self.nb_class, feature_extractor=feature_extractor)

        
    def custom_loss(self, y_true, y_pred):
        return 

    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    valid_times,    # the number of times to repeat the validation set, often used for small datasets
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    saved_weights_name='best_weights.h5',
                    debug=False):  

        self.batch_size = batch_size

        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale

        self.debug = debug

        ############################################
        # Make train and validation generators
        ############################################

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }  

        dataloader = data_generator(train_imgs, generator_config, norm=self.Yolo.normalize)
                  
        x_batch, b_batch, y_batch = dataloader[0]

        print(x_batch.shape, b_batch.shape, y_batch.shape)

        plt.imshow(x_batch[0, :, :])
        plt.show()

        """
        train_generator = DataLoader(dataloader, shuffle=True, batch_size=10)
   
        for x_batch, b_batch, y_batch in train_generator:
             print(x_batch.shape, b_batch.shape, y_batch.shape)
        
        train_generator = BatchGenerator(train_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize)
        valid_generator = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=self.feature_extractor.normalize,
                                     jitter=False)   
        """
    
        
        



