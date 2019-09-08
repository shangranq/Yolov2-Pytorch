import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from util import decode_netout, compute_overlap, compute_ap, draw_boxes_object
from backend import Yolo_v2
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
        self.Yolo = self.Yolo.cuda()

    def custom_loss(self, y_pred, y_true, true_boxes):
        
        # y_true and y_pred shape (batch, box, 5+nb_class, grid_h, grid_w)
        # true_boxes shape (batch, 1, 1, 1, max_boxes_per_image, 4)
        # print(y_true.shape, y_pred.shape, true_boxes.shape)
        # swap axies from channel first to channel last
        y_true = y_true.permute(0, 3, 4, 1, 2)
        y_pred = y_pred.permute(0, 3, 4, 1, 2)

        # print(y_true.shape, y_pred.shape, true_boxes.shape)
        # y_true and y_pred shape (batch, grid_h, grid_w, box, 5+nb_class)
        # true_boxes shape (batch, 1, 1, 1, max_boxes_per_image, 4)
        
        # cell_x shape (1, grid_h, grid_w, 1, 1) where cell_x[0, a, b, 0, 0] = b
        # cell_y shape (1, grid_h, grid_w, 1, 1) where cell_y[0, a, b, 0, 0] = a
        cell_x = np.reshape(np.tile(np.arange(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1))
        cell_y = np.transpose(cell_x, (0,2,1,3,4))
        # cell_grid shape (batch, grid_h, grid_w, nb_box, 2)
        cell_grid = torch.from_numpy(np.tile(np.concatenate([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1]))
        # print(cell_grid.shape)
        # print(cell_grid[0, 4, 6, 0, :]==[6, 4])

        """
        Adjust prediction
        """
        ### adjust x and y 
        # pred_box_xy shape (batch, grid_h, grid_w, box, 2)
        # sigmoid(y_pred_xy) returns the predicted location relative to the cell (one assumption is the cell length is 1)
        # adding cell_grid makes it the location relative to the whole image
        pred_box_xy = torch.sigmoid(y_pred[..., :2]) + cell_grid.cuda().float()

        ### adjust w and h
        # predicted bounding box height = anchor_box_h * exp(y_pred_h) 
        # predicted bounding box width = anchor_box_w * exp(y_pred_w)
        pred_box_wh = torch.exp(y_pred[..., 2:4]) * torch.from_numpy(np.reshape(self.anchors, [1,1,1,self.nb_box,2])).cuda().float()

        ### adjust confidence
        # confidence ranges from 0 to 1
        pred_box_conf = torch.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        # true_box_xy: ground truth box center xy relative to the upper left corner of the image
        # in the grid scale, thus true_box_xy ranges from (0, grid_h) (0, grid_w)
        true_box_xy = y_true[..., 0:2]  
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] 
        
        ### adjust confidence based on IOU 
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half    # shape (batch, grid_h, grid_w, box, 2)
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half    # shape (batch, grid_h, grid_w, box, 2) 
        
        intersect_mins  = torch.max(pred_mins,  true_mins)
        intersect_maxes = torch.min(pred_maxes, true_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros(intersect_maxes.size()).cuda())
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]   # shape (batch, grid_h, grid_w, box, 1)
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas   
        iou_scores  = torch.div(intersect_areas, union_areas)           # shape (batch, grid_h, grid_w, box, 1)
        # the iou_scores calculated all IOU between predicted boxes and true boxes that were assigned to that cell
        
        # y_true[..., 4] is a mask whose value is either 0 (no objects in the cell) or 1 (object in the cell)
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        _, true_box_class = torch.max(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        # shape (batch, grid_h, grid_w, box, 1)
        coord_mask = y_true[..., 4].unsqueeze(-1) * self.coord_scale 

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        # shape (batch, 1, 1, 1, max_number_of_boxes_per_image, 2)
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy.view(self.batch_size, self.grid_h, self.grid_w, self.nb_box, 1, 2)
        pred_wh = pred_box_wh.view(self.batch_size, self.grid_h, self.grid_w, self.nb_box, 1, 2)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half       

        # true_mins shape (batch, 1, 1, 1, max_number_of_boxes_per_image, 2)
        # pred_mins shape (batch, grid_h, grid_w, nb_box, 1, 2)
        intersect_mins  = torch.max(pred_mins,  true_mins)
        # intersect_mins shape (batch, grid_h, grid_w, nb_box, max_number_of_boxes_per_image, 2)
        intersect_maxes = torch.max(pred_maxes, true_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros(intersect_maxes.size()).cuda())
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  
        # intersect_areas shape (batch, grid_h, grid_w, nb_box, max_number_of_boxes_per_image)
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = torch.div(intersect_areas, union_areas)

        best_ious, _ = torch.max(iou_scores, dim=4)
        # best_ious shape (batch, grid_h, grid_w, nb_box)
        conf_mask = torch.lt(best_ious, 0.6).float() * torch.eq(y_true[..., 4], 0).float() * self.no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        # class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale
        # need to figure out what is tf.gather 
        # class_mask = y_true[..., 4] * true_box_class.float() * self.class_scale # this is a bug!
        class_mask = y_true[..., 4] * self.class_scale
 
        nb_coord_box = torch.sum(torch.gt(coord_mask, 0.0).float())
        nb_conf_box  = torch.sum(torch.gt(conf_mask, 0.0).float())
        nb_class_box = torch.sum(torch.gt(class_mask, 0.0).float())

        # wrong print(nb_coord_box, nb_conf_box, nb_class_box)
        
        loss_xy    = torch.sum((true_box_xy-pred_box_xy)**2     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = torch.sum((true_box_wh-pred_box_wh)**2     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = torch.sum((true_box_conf-pred_box_conf)**2 * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        # loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        # loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        # pred_box_class swap axes 16, 13, 13, 5, 80 into 16, 80, 13, 13, 5
        # true_box_class 16, 13, 13, 5
        pred_box_class = pred_box_class.permute(0, 4, 1, 2, 3)
        loss_class = nn.CrossEntropyLoss(reduction='none')(pred_box_class, true_box_class)     
        loss_class = (loss_class * class_mask).sum() / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        """
        Debugging code
        """
        self.batch_index += 1

        if self.debug and self.batch_index % 1500 == 0:
            nb_true_box = torch.sum(y_true[..., 4])
            nb_pred_box = torch.sum(torch.gt(true_box_conf, 0.5).float() * torch.gt(pred_box_conf, 0.3).float())
            current_recall = nb_pred_box.data / (nb_true_box.data + 1e-6)
            print('batch index ', self.batch_index, '/ epoch index ', self.epoch_index)
            print('loss_xy: ', loss_xy)
            print('loss_wh: ', loss_wh)
            print('loss_conf: ', loss_conf)
            print('loss_class: ', loss_class)
            print('loss: ', loss)
            print('current_recall: ', current_recall)
            print('*' * 80)

        return loss
    
    def load_weights(self, weight_path):
        self.Yolo.load_state_dict(torch.load(weight_path))     
    
    def train(self, train_imgs,     # the list of images to train the model
                    valid_imgs,     # the list of images used to validate the model after each epoch
                    test_imgs,      # the list of images used to test the model at the end of training 
                    nb_epochs,      # number of epoches
                    learning_rate,
                    batch_size,
                    object_scale,
                    no_object_scale,
                    coord_scale,
                    class_scale,
                    saved_weights_name,
                    train_last,
                    debug):

        self.batch_size = batch_size
        self.object_scale    = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale     = coord_scale
        self.class_scale     = class_scale
        self.debug = debug
        self.train_last = train_last
        self.batch_index = 0
        self.epoch_index = 0

        if self.train_last:
            for param in self.Yolo.feature_extractor.parameters():
                param.requires_grad = False
            optimizer = optim.Adam(self.Yolo.detect_layer.parameters(), lr=learning_rate, betas=[0.9, 0.999], eps=1e-08,
                                   weight_decay=0.0)
        else:
            for param in self.Yolo.feature_extractor.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(self.Yolo.parameters(), lr=learning_rate, betas=[0.9, 0.999], eps=1e-08,
                                   weight_decay=0.0)

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

        train_dataloader = data_generator(train_imgs, generator_config, norm=self.Yolo.normalize)
        valid_dataloader = data_generator(valid_imgs, generator_config, norm=self.Yolo.normalize, jitter=False)
        train_batch_generator = DataLoader(train_dataloader, num_workers=8, drop_last=True, shuffle=True, batch_size=generator_config['BATCH_SIZE'])
        valid_batch_generator = DataLoader(valid_dataloader, num_workers=8, drop_last=True, shuffle=False, batch_size=generator_config['BATCH_SIZE'])

        ############################################
        # Start the training process
        ############################################
        val_best_loss = float('inf')

        print('validating the model ...')
        val_loss = self.val_epoch(self.Yolo, valid_batch_generator, self.custom_loss)
        print('validation loss is', val_loss)
        self.evaluate(valid_imgs[:1000])
        self.batch_index = 0
        self.epoch_index = 0

        for epoch in range(nb_epochs):
            print('*'*40)
            print('training the {} th epoch ...'.format(epoch))
            self.train_epoch(self.Yolo, train_batch_generator, optimizer, self.custom_loss)
            print('validating the model ...')
            val_loss = self.val_epoch(self.Yolo, valid_batch_generator, self.custom_loss)
            print('validation loss is', val_loss)
            self.evaluate(valid_imgs[:1000])

            if val_loss < val_best_loss:
                val_best_loss = val_loss
                torch.save(self.Yolo.state_dict(), saved_weights_name)
            self.epoch_index += 1
            self.batch_index = 0
                                
        ############################################
        # Compute mAP on the validation set
        ############################################
        print('evaluating the testing set mAP ... ')
        average_precisions = self.evaluate(test_imgs)

    def train_epoch(self, model,
                          dataloader,
                          optimizer,
                          criterion):
        model.train(True)
        for x_batch, b_batch, y_batch in dataloader:
            x_batch, b_batch, y_batch = x_batch.cuda().float(), b_batch.cuda().float(), y_batch.cuda().float()
            model.zero_grad()
            y_batch_pred = model(x_batch)
            loss = criterion(y_batch_pred, y_batch, b_batch)
            loss.backward()
            optimizer.step()
            
    def val_epoch(self, model,
                        dataloader,
                        criterion):
        model.train(False)
        loss = 0
        with torch.no_grad():
            for x_batch, b_batch, y_batch in dataloader:
                x_batch, b_batch, y_batch = x_batch.cuda().float(), b_batch.cuda().float(), y_batch.cuda().float()
                y_batch_pred = model(x_batch)
                loss += criterion(y_batch_pred, y_batch, b_batch).data.cpu()
        return loss
    
    def evaluate(self, 
                 test_imgs, 
                 iou_threshold=0.3,
                 obj_threshold=0.2,
                 nms_threshold=0.3):

        self.Yolo.train(False)

        generator_config = {
            'IMAGE_H'         : self.input_size,
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : 1,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }

        # evaluation has to be in the eval stage
  
        generator = data_generator(test_imgs, generator_config, norm=self.Yolo.normalize, jitter=False)

        # gather all detections and annotations
        all_detections = [[None for i in range(generator.num_classes())] for j in range(len(generator))]
        all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

        for i in range(len(generator)):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image, obj_threshold, nms_threshold)

            if i < 40:
                image_bbox = draw_boxes_object(raw_image, pred_boxes, self.labels)
                cv2.imwrite('./sample/image_pred_box/test_pred_{}.png'.format(i), image_bbox)

            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(len(generator)):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        # for label, average_precision in average_precisions.items():
        #     print(self.labels[label], '{:.4f}'.format(average_precision))
        print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

        return average_precisions

    def predict(self, image, obj_threshold, nms_threshold):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.Yolo.normalize(image)

        input_image = image[:,:,::-1].transpose(2, 0, 1)
        input_image = torch.Tensor(np.expand_dims(input_image, 0).copy()).cuda()

        netout = self.Yolo(input_image)
        boxes  = decode_netout(netout.data.cpu().numpy(), self.anchors, self.nb_class, obj_threshold=obj_threshold, nms_threshold=nms_threshold)

        return boxes
    
            
            
     

        
    
        
        



