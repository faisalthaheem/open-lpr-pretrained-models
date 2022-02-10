#!/bin/sh

# Get COCO 2017 data sets
mkdir coco
pushd coco

curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip

popd

echo "Coco 2017 dataset downloaded"