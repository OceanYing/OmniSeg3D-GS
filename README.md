# OmniSeg3D-GS: Gaussian-Splatting based OmniSeg3D (CVPR2024)

### [Project Page](https://oceanying.github.io/OmniSeg3D/) | [Arxiv Paper](https://arxiv.org/abs/2311.11666)

[OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning](https://arxiv.org/abs/2311.11666)  
[Haiyang Ying](https://oceanying.github.io/)<sup>1</sup>, Yixuan Yin<sup>1</sup>, Jinzhi Zhang<sup>1</sup>, Fan Wang<sup>2</sup>, Tao Yu<sup>1</sup>, Ruqi Huang<sup>1</sup>, [Lu Fang](http://www.luvision.net/)<sup>1</sup>   
<sup>1</sup>Tsinghua Univeristy &emsp; <sup>2</sup>Alibaba Group.  

OmniSeg3D is a framework for multi-object, category-agnostic, and hierarchical segmentation in 3D, the [original implementation](https://github.com/THU-luvision/OmniSeg3D) is based on InstantNGP.

However, OmniSeg3D is not restricted by specific 3D representation. In this repo, we present a guassian-splatting based OmniSeg3D, which enjoys interactive 3D segmentation in real-time. The segmented objects can be saved as .ply format for further visualization and manipulation.

![image](https://github.com/OceanYing/OmniSeg3D-GS/assets/37448328/60cb1019-5734-4c25-a51e-6b43b2bcd4db)


## Installation
We follow the original environment setting of [3D Guassian-Splatting (SIGGRAPH 2023)](https://github.com/graphdeco-inria/gaussian-splatting).

```shell
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Install `SAM` for 2D segmentation:
```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
mkdir sam_ckpt; cd sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Data Preparation:
We typically support data prepared as COLMAP format. For more details, please refer to the [guidance](https://github.com/THU-luvision/OmniSeg3D#hierarchical-representation-generation) in our [NeRF-based implementation of OmniSeg3D](https://github.com/THU-luvision/OmniSeg3D).

### Hierarchical Representation Generation
Run the sam model to get the hierarchical representation files.
```bash
python run_sam.py --ckpt_path {SAM_CKPT_PATH} --file_path {IMAGE_FOLDER}
```
After running, you will get three folders `sam`, `masks`, `patches`:
* `sam`: stores the hierarchical representation as ".npz" files
* `masks` and `patches`: used for visualization or masks quaility evaluation, not needed during training.

Ideal `masks` should include object-level masks and `patches` should contain part-level masks. We basically use the default parameter setting for SAM, but you can tune the parameters for customized datasets.


## Training:
We train our models on a sinle NVIDIA RTX 3090 Ti GPU (24GB). Smaller scenes may require less memory. Typically, inference requires less than 8GB memory.
We utilize a two-stage training strategy. See script/train_omni_360.sh as an example.
```bash
dataname=counter
gpu=1
data_path=root_path/to/the/data/folder/of/counter.

# --- Training Gaussian (Color and Density) --- #
CUDA_VISIBLE_DEVICES=${gpu} python train.py \
     -s ${data_path} \
     --images images_4 \
     -r 1 -m output/360_${dataname}_omni_1/rgb \
     --config_file config/gaussian_dataset/train_rgb.json \
     --object_path sam \
     --ip 127.0.0.2

# --- Training Semantic Feature Field --- #
CUDA_VISIBLE_DEVICES=${gpu} python train.py \
     -s ${data_path} \
     --images images_4 \
     -r 1 \
     -m output/360_${dataname}_omni_1/sem_hi \
     --config_file config/gaussian_dataset/train_sem.json \
     --object_path sam \
     --start_checkpoint output/360_${dataname}_omni_1/rgb/chkpnt10000.pth \
     --ip 127.0.0.2

# --- Render Views for Visualization --- #
CUDA_VISIBLE_DEVICES=${gpu} python render_omni.py \
    -m output/360_${dataname}_omni_1/sem_hi \
    --num_classes 256 \
    --images images_4
```
After specifying the custom information, you can run the file by execute at the root folder:
```bash
bash script/train_omni_360.sh
```

## GUI Visualization and Segmentation

Modify the path of the trained point cloud. Then run ``render_omni_gui.py``.

![Screenshot 2024-03-25 21:20:50 - omniseg3dgs](https://github.com/OceanYing/OmniSeg3D-GS/assets/37448328/47912c9d-16ac-48fc-9d05-23bd1f83a333)
![Screenshot 2024-03-25 21:21:08 - omniseg3dgs](https://github.com/OceanYing/OmniSeg3D-GS/assets/37448328/60c4a026-77ed-4dc2-85e2-587ca134e2a2)
![Screenshot 2024-03-25 21:21:54 - omniseg3dgs](https://github.com/OceanYing/OmniSeg3D-GS/assets/37448328/9e0b0898-0602-41c6-a581-c6d3197e1eed)


### GUI options:
- ``mode option``: RGB, score map, and semantic map (you can visualize the consistent global semantic feature).
- ``click mode``: select object of interest
- ``multi-click mode``: select multiple points or objects
- ``binary threshold``: show binarized 2D images with the threshold
- ``segment3d``: segment the scene with the current threshold (saved .ply file can be found at the root dir)
- ``reload``: reload the whole scene
- ``file selector``: load another scene (point cloud)

### Operations:
- ``left drag``: rotate
- ``mid drag``: pan
- ``right click``: choose point/objects


## Acknowledgements
Thanks for the following project for their valuable contributions:
- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Gaussian-Grouping](https://github.com/lkeab/gaussian-grouping)


## Citation
If you find this project helpful for your research, please consider citing the report and giving a ⭐.
```BibTex
@article{ying2023omniseg3d,
  title={OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning},
  author={Ying, Haiyang and Yin, Yixuan and Zhang, Jinzhi and Wang, Fan and Yu, Tao and Huang, Ruqi and Fang, Lu},
  journal={arXiv preprint arXiv:2311.11666},
  year={2023}
}
```
