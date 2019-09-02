import argparse
import os
import numpy as np
from util import parse_annotation
from frontend import YOLO
import json
import torch

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on COCO 2014 dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

def _main_(args):

    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    test_imgs, test_labels = parse_annotation(config['test']['test_annot_folder'],
                                              config['test']['test_image_folder'],
                                              config['model']['labels'])

    ###############################
    #   Construct the model 
    ###############################
    yolo = YOLO(feature_extractor   = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors']) 

    yolo.load_weights(config['train']['saved_weights_name'])
    print('weights loaded from {}'.format(config['train']['saved_weights_name']))

    yolo.evaluate(test_imgs, iou_threshold=0.3, obj_threshold=0.4, nms_threshold=0.2)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

