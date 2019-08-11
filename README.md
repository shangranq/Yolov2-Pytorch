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
convert COCO annotation format to Pascal VOC format, run: 
```
mkdir train2014_annotations/ 
python coco2pascal.py create_annotations COCO/ train COCO/train2014_annotations/ 
mkdir val2014_annotations/
python coco2pascal.py create_annotations COCO/ val COCO/val2014_annotations/  
```



