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
        if [ $i -lt 8000 ]
        then
            echo 'GPU' $count ' is avaiable, start training...'
            CUDA_VISIBLE_DEVICES=$count \
            nohup python train.py \
            --dataset_name 'ICFG-PEDES' \
            --root_dir '/data0/data_ccq/ICFG/' \
            --output_dir '/data1/ccq/multimodality-ICFG' \
            --img_aug \
            --name 'sketch2_add-fusion-twofocal-1-35-fusion-itcloss_03kl-text-label' \
            --fusion_way 'add' \
            --batch_size 64 \
            --pa 0.1 \
            --pretrain_choice 'ViT-B/16' \
            --loss_names 'itc' \
            --lrscheduler 'cosine' \
            --target_lr 0 \
            --num_epoch 60 \
            --al 1.0 \
            --ga 3.5 \
            --klp 0.3 \
            --focal_three_fusion_loss3 \
            > scripts/ICFG-PEDES/ViT/nohup2.out &
            _pid=$!
            echo "training pid: $_pid"
            flag=1
            break
        fi
        count=$(($count+1))    
    done
    sleep 20
done

#--name $name \
# --root_dir '/data0/data_ccq/CUHK-PEDES/' \