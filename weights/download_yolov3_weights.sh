#!/bin/bash

mkdir -p weights && cd weights

wget -c https://pjreddie.com/media/files/yolov3.weights
wget -c https://pjreddie.com/media/files/yolov3-tiny.weights
wget -c https://pjreddie.com/media/files/yolov3-spp.weights
 
wget -c https://pjreddie.com/media/files/darknet53.conv.74

 