dataname=lego
gpu=1
datapath=/data/haiyang/projects/Datasets/nerf_synthetic_data/${dataname}

CUDA_VISIBLE_DEVICES=${gpu} python train.py \
     -s ${datapath} \
     -r 1 -m output/blender_${dataname}_omni_1/rgb \
     --config_file config/gaussian_dataset/train_rgb.json \
     --object_path sam \
     --ip 127.0.0.1


CUDA_VISIBLE_DEVICES=${gpu} python train.py \
     -s ${datapath} \
     -r 1 \
     -m output/blender_${dataname}_omni_1/sem_hi \
     --config_file config/gaussian_dataset/train_sem.json \
     --object_path sam \
     --train_split \
     --start_checkpoint output/blender_${dataname}_omni_1/rgb/chkpnt10000.pth \
     --ip 127.0.0.1


CUDA_VISIBLE_DEVICES=${gpu} python render_omni.py \
    -m output/blender_${dataname}_omni_1/sem_hi \
    --num_classes 256 \
    --images images
