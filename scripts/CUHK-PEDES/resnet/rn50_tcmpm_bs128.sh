#!/bin/bash
flag=0
file=${0%%.*}
name=${file##*/}
# time=$(date +%Y%m%d%H%M%S)
while [ $flag -eq 0 ]
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt 500 ]
        then
            echo 'GPU' $count ' is avaiable, start training...'
            CUDA_VISIBLE_DEVICES=$count \
            nohup python train.py \
            --name $name \
            --img_aug \
            --batch_size 128 \
            --pretrain_choice 'RN50' \
            --loss_names 'tcmpm' \
            --lrscheduler 'cosine' \
            --target_lr 0 \
            --num_epoch 60 \
            > scripts/nohups/resnet/$name.out &
            _pid=$!
            echo "training pid: $_pid"
            flag=1
            break
        fi
        count=$(($count+1))    
    done
    sleep 20
done