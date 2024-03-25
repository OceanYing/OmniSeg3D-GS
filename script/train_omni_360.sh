dataname=counter
gpu=1
data_path=/data/haiyang/projects/Datasets/360/${dataname}

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#      -s ${data_path} \
#      --images images_4 \
#      -r 1 -m output/360_${dataname}_omni_1/rgb \
#      --config_file config/gaussian_dataset/train_rgb.json \
#      --object_path sam \
#      --ip 127.0.0.2


CUDA_VISIBLE_DEVICES=${gpu} python train.py \
     -s ${data_path} \
     --images images_4 \
     -r 1 \
     -m output/360_${dataname}_omni_1/sem_hi \
     --config_file config/gaussian_dataset/train_sem.json \
     --object_path sam \
     --start_checkpoint output/360_${dataname}_omni_1/rgb/chkpnt10000.pth \
     --ip 127.0.0.2


CUDA_VISIBLE_DEVICES=${gpu} python render_omni.py \
    -m output/360_${dataname}_omni_1/sem_hi \
    --num_classes 256 \
    --images images_4
