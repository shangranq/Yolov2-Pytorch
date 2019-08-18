from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle
import cv2
import copy
import imgaug as ia
from imgaug import augmenters as iaa

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c     = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score
    

class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):
        '''
        ANCHORS: a np.array of even number length e.g.
        
        _ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
                    2,4, ##  width=2, height=4,  tall large anchor box
                    1,1] ##  width=1, height=1,  small anchor box
        '''
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
                        for i in range(int(len(ANCHORS)//2))]
        
    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union
    
    def find(self, center_w, center_h):
        # find the anchor that has highest IOU with this ground truth box
        best_anchor = -1
        max_iou     = -1
        shifted_box = BoundBox(0, 0, center_w, center_h)
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou = iou
        return best_anchor, max_iou
    
    
def rescale_centerxy(obj, config):
    '''
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    '''
    center_x = .5*(obj['xmin'] + obj['xmax'])
    center_x = center_x / (float(config['IMAGE_W']) / config['GRID_W'])
    center_y = .5*(obj['ymin'] + obj['ymax'])
    center_y = center_y / (float(config['IMAGE_H']) / config['GRID_H'])
    return center_x, center_y


def rescale_cebterwh(obj, config):
    '''
    obj:     dictionary containing xmin, xmax, ymin, ymax
    config : dictionary containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
    '''    
    center_w = (obj['xmax'] - obj['xmin']) / (float(config['IMAGE_W']) / config['GRID_W']) 
    center_h = (obj['ymax'] - obj['ymin']) / (float(config['IMAGE_H']) / config['GRID_H']) 
    return center_w, center_h


class data_generator(Dataset):
    def __init__(self, images, config, jitter=True, norm=None):
        self.images = images
        self.config = config
        self.norm = norm
        self.jitter = jitter
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['ANCHORS'])
        
        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug_pipe = iaa.Sequential(
            [
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        ]), 
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        

    def __len__(self):
        return len(self.images)
    
    def num_classes(self):
        return len(self.config['LABELS'])
    
    def load_annotation(self, i):
        annots = []
        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]
        if len(annots) == 0: annots = [[]]
        return np.array(annots)
    
    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        
        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
            
            ### change brightness, sharpness, blurness, dropout and so on .. pixel value augmentation 
            image = self.aug_pipe.augment_image(image)            
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]  # BGR to RGB

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
                
        return image, all_objs

    def __getitem__(self, idx):
        """
        for a single image: 
            x_batch (3, 416, 416)
            b_batch (true_box_buffer, 4)
            y_batch (5+nb_class, 13, 13)
        """
        x_batch = np.zeros((3, self.config['IMAGE_H'], self.config['IMAGE_W']))                      
        b_batch = np.zeros((1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))  
        y_batch = np.zeros((self.config['BOX'], 4+1+len(self.config['LABELS']), self.config['GRID_H'], self.config['GRID_W']))     

        img = self.images[idx]
        
        img, all_objs = self.aug_image(img, jitter=self.jitter)
        
        true_box_index = 0
        
        for obj in all_objs:

            if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                center_x, center_y = rescale_centerxy(obj, self.config)
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
        
                if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    obj_indx  = self.config['LABELS'].index(obj['name'])
                    center_w, center_h = rescale_cebterwh(obj, self.config)
                    box = [center_x, center_y, center_w, center_h]
                    best_anchor, max_iou = self.bestAnchorBoxFinder.find(center_w, center_h)
                
                    y_batch[best_anchor, 0:4, grid_y, grid_x] = box
                    y_batch[best_anchor, 4, grid_y, grid_x] = 1. 
                    y_batch[best_anchor, 5+obj_indx, grid_y, grid_x] = 1 
                                
                    b_batch[0, 0, 0, true_box_index] = box        
                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                
        img = self.norm(img) if self.norm != None else img
        x_batch = np.transpose(img, (2, 0, 1))  # channel last to channel first

        return x_batch, b_batch, y_batch            
                

if __name__ == "__main__":
    pass






