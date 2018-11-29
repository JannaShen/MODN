#!/usr/bin/env sh

LOG=/data/jishen/MODN/log.txt
CAFFE=/home/jishen/my_caffe_gpu/caffe-master/build/tools/caffe
#your caffe path
$CAFFE train --solver=/data/jishen/MODN/caffe-files/train_server.prototxt  -gpu 0 2>&1 | tee $LOG

