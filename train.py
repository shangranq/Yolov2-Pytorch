import argparse
import os
import numpy as np
from util import parse_annotation
from frontend import YOLO
import json
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1"

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

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

    ###############################
    #   Parse the annotations 
    ###############################

    """
    parse annotations of the training set
    train_imgs is a list of dictionary with the following keys and contents: 
        object   : a list of objects in the image, each object is a dictionary with object name, box coordinates (xmin, xmax, ymin, ymax)
        filename : complete path of the image  
        width    : original pixel width
        height   : original pixel height
    train_labels contains the statistics of the count of each type of object 
    """

    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'], 
                                                config['train']['train_image_folder'], 
                                                config['model']['labels'])
 
    # split the training set into 80% training and 20% validation
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)
        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]
        print('{} train images and {} validation images'.format(len(train_imgs), len(valid_imgs)))

    # parse annotations of the testing set
    test_imgs, test_labels = parse_annotation(config['test']['test_annot_folder'],
                                              config['test']['test_image_folder'],
                                              config['model']['labels'])
    
    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', len(train_labels))
        print('Given labels:\t', len(config['model']['labels']))
        print('Overlap labels:\t', len(overlap_labels))
        print(overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        print(train_labels.keys())
        config['model']['labels'] = train_labels.keys()

    ###############################
    #   Construct the model 
    ###############################
    yolo = YOLO(feature_extractor   = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ##############################################################
    #   Start training the last layer from scratch with warm up
    ##############################################################
    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               test_imgs          = test_imgs,
               pretrained_weights = '',
               nb_epochs          = config['train']['nb_epochs'],
               learning_rate      = config['train']['learning_rate'],
               batch_size         = config['train']['batch_size'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               train_last_epoch   = 3,
               freeze_BN          = True,
               train_mode         = False,
               debug              = True)



if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

