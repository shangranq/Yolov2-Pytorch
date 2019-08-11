# Project
Pytorch version of the experiencor's keras YOLOv2 implementation https://github.com/experiencor/keras-yolo2 

## Getting Started

### Prepare the COCO dataset:
create a folder for COCO dataset 
```
mkdir COCO/
cd COCO/
```
download the data from the COCO website by running: 
```
wget http://images.cocodataset.org/zips/train2014.zip 
wget http://images.cocodataset.org/zips/val2014.zip 
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip 
```
unzip all above zip files:
```
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip 
```
create two subfolders in the COCO/ folder to store Pascal VOC format annotations:
```
mkdir train2014_annotations/ 
mkdir val2014_annotations/
```
git clone this repo to your computer and use the coco2pascal.py script to convert COCO annotation format to Pascal VOC format 
```
git clone https://github.com/shangranq/Yolov2-Pytorch.git
cd Yolov2_Pytorch/
python coco2pascal.py create_annotations COCO/ train COCO/train2014_annotations/ 
python coco2pascal.py create_annotations COCO/ val COCO/val2014_annotations/  
```

So far the dataset has been prepared and the data folder structure should be the same as:
```
.
    ├── COCO
          ├──train2014                    # training set images
          ├──val2014                      # validation set images
          ├──train2014_annotations        # Pascal VOC format training set annotation
          ├──val2014_annotations          # Pascal VOC format validation set annotation 
          ├──annotations                  # original COCO annotation 
    
``` 






