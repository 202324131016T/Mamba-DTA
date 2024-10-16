#!/bin/bash

#cd /data/wyl/CoVAE-master/
#conda activate CoVAE

datatime=$(date +"%Y%m%d-%H%M%S")
# train
#CUDA_VISIBLE_DEVICES=1 \
#python run_experiments.py \
#--log_dir ./logs/${datatime}/ \
#--checkpoint_path ./checkpoints/${datatime}/ \
#--result ./result/${datatime}/ \
#> ./logs/train_${datatime}_run_experiments.txt # 日志输出路径


#datatime=20231230-144959
## only test
CUDA_VISIBLE_DEVICES=0 \
python run_experiments.py \
--log_dir ./logs/"${datatime}_test/" \
--checkpoint_path ./checkpoints/${datatime}/ \
--result ./result/"${datatime}_test/" \
--only_test True \
> ./logs/"test_${datatime}_run_experiments.txt" # 日志输出路径
