This is a Pytorch implementation of YOLOv2 that is inspired by experiencor's keras YOLOv2 implementation. 

Download COCO dataset:
create a folder for COCO dataset (i.e. mkdir COCO/), in the COCO/ folder, run: \
wget http://images.cocodataset.org/zips/train2014.zip \
wget http://images.cocodataset.org/zips/val2014.zip \
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip \
unzip all above zip files 

TO convert COCO annotation format to Pascal VOC format, run: \
mkdir new_annotations/ \
python coco2pascal.py create_annotations COCO/ train COCO/new_annotations/ \
python coco2pascal.py create_annotations COCO/ val COCO/new_annotations/  


