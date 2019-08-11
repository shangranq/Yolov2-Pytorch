from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle
import cv2
import copy

class ImageReader(object):
    def __init__(self,IMAGE_H,IMAGE_W, norm=None):
        # IMAGE_H and IMAGE_W is the standard size of input from the config file
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.norm    = norm
        
    def encode_core(self,image, reorder_rgb=True):     
        image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))
        if reorder_rgb:
            image = image[:,:,::-1]
        if self.norm is not None:
            image = self.norm(image)
        image = np.transpose(image, (2, 0, 1)) # pytorch is channel first 
        return image
    
    def fit(self, train_instance):
        '''
        read in and resize the image, annotations are resized accordingly.
        
        -- Input -- 
        
        train_instance : dictionary containing filename, height, width and object
        
        {'filename': 'ObjectDetectionRCNN/VOCdevkit/VOC2012/JPEGImages/2008_000054.jpg',
         'height':   333,
         'width':    500,
         'object': [{'name': 'bird',
                     'xmax': 318,
                     'xmin': 284,
                     'ymax': 184,
                     'ymin': 100},
                    {'name': 'bird', 
                     'xmax': 198, 
                     'xmin': 112, 
                     'ymax': 209, 
                     'ymin': 146}]
        }
        
        '''
        if not isinstance(train_instance, dict):
            train_instance = {'filename': train_instance}
                
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        h, w, c = image.shape
        if image is None: print('Cannot find ', image_name)
        # resize the image to standard size
        image = self.encode_core(image, reorder_rgb=True)
        if "object" in train_instance.keys():
            all_objs = copy.deepcopy(train_instance['object'])     
            # rescale the xmin, xmax, ymin, ymax of all objects accordingly 
            for obj in all_objs:
                for attr in ['xmin', 'xmax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_W) / w)
                    obj[attr] = max(min(obj[attr], self.IMAGE_W), 0)
                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.IMAGE_H) / h)
                    obj[attr] = max(min(obj[attr], self.IMAGE_H), 0)
        else:
            return image
        return image, all_objs


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
        shifted_box = BoundBox(0, 0,center_w, center_h)
        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
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


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None,classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        self.confidence = confidence
        self.set_class(classes)
        
    def set_class(self,classes):
        self.classes = classes
        self.label   = np.argmax(self.classes) 
        
    def get_label(self):  
        return(self.label)
    
    def get_score(self):
        return(self.classes[self.label])


class data_generator(Dataset):

    def __init__(self, imgs, config, norm):
        self.imgs = imgs
        self.config = config
        self.norm = norm
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(config['ANCHORS'])
        self.imageReader = ImageReader(config['IMAGE_H'],config['IMAGE_W'],norm=norm) 

    def __len__(self):
        return len(self.imgs)

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

        img = self.imgs[idx]
        img, all_objs = self.imageReader.fit(img)
        true_box_index = 0

        for obj in all_objs:

            if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                center_x, center_y = rescale_centerxy(obj,self.config)
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
        
            if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                obj_indx  = self.config['LABELS'].index(obj['name'])
                center_w, center_h = rescale_cebterwh(obj,self.config)
                box = [center_x, center_y, center_w, center_h]
                best_anchor,max_iou = self.bestAnchorBoxFinder.find(center_w, center_h)
                
                y_batch[best_anchor, 0:4, grid_y, grid_x] = box
                y_batch[best_anchor, 4, grid_y, grid_x] = 1. 
                y_batch[best_anchor, 5+obj_indx, grid_y, grid_x] = 1 
                                
                b_batch[0, 0, 0, true_box_index] = box        
                true_box_index += 1
                true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

        x_batch = img

        return x_batch, b_batch, y_batch            
                

if __name__ == "__main__":
    pass






