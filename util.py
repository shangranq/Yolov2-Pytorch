import xml.etree.ElementTree as ET
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class BoundBox:
    def __init__(self, xmin, xmax, ymin, ymax, obj_type):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.obj_type = obj_type 


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in tqdm(sorted(os.listdir(ann_dir))):
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels


def load_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def plot_anchor(xmin, xmax, ymin, ymax, iobj, w):
    color_palette = list(sns.xkcd_rgb.values())
    c = color_palette[iobj]
    plt.plot(np.array([xmin,xmin]), np.array([ymin,ymax]), color=c, linewidth=w)
    plt.plot(np.array([xmin,xmax]), np.array([ymin,ymin]), color=c, linewidth=w)
    plt.plot(np.array([xmax,xmax]), np.array([ymax,ymin]), color=c, linewidth=w)
    plt.plot(np.array([xmin,xmax]), np.array([ymax,ymax]), color=c, linewidth=w)

def draw_boxes(image, xmin, xmax, ymin, ymax, label):
    image_h, image_w, _ = image.shape
    a, b, c = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (a, b, c), 3)
    cv2.putText(image, label, 
               (xmin, ymin+20), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               1e-3 * image_h, 
               (a,b,c), 2)
    return image    
                   
def plot_image_anchors(index):
    image = train_image[index]
    image_path = image['filename']
    objects = image['object']
    image = load_image(image_path)
    for iobj, obj in enumerate(objects):
        x_min, x_max, y_min, y_max = obj['xmin'], obj['xmax'], obj['ymin'], obj['ymax']
        image = draw_boxes(image, x_min, x_max, y_min, y_max, obj['name'])
    plt.imshow(image)
    plt.show()


if __name__ == '__main__': 
    ##annotations
    train_annot_folder = '/data/datasets/COCO/new_annotations/'
    train_image_folder = '/data/datasets/COCO/train2014/' 
    train_image = parse_annotation(train_annot_folder, train_image_folder)
    print("N train = {}".format(len(train_image)))
    print(train_image[0])
    for i in range(60, 61):
        plot_image_anchors(0)

