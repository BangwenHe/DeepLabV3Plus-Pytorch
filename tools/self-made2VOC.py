"""
将自己拍摄的数据集修改为类似VOC格式的数据集。

原数据集格式: 
self_made_cdd
├─ images
│  ├─ video1
│  │  ├─ frame_00001.png
│  │  └─ ...
│  ├─ video2
│  └─ ...
├─ mvs_256
│  ├─ video1.txt
│  ├─ video2.txt
│  └─ ...
└─ label
   ├─ video1
   │  └─ pesudo_color
   │     ├─ frame_00001.png
   │     └─ ...
   ├─ video2
   └─ ...

转换后的数据集格式: 
VOCdevkit
└─ VOC2012
   ├─ CompressedDomainData
   │  ├─ video1_frame00000.txt
   │  ├─ video1_frame00001.txt
   │  └─ ...
   ├─ ImageSets
   │  └─ Segmentation
   │     ├─ train.txt
   │     └─ val.txt
   ├─ JPEGImages
   │  ├─ video1_frame00000.jpg
   │  ├─ video1_frame00001.jpg
   │  └─ ...
   └─ SegmentationClass
      ├─ video1_frame00000.png
      ├─ video1_frame00001.png
      └─ ...
"""

import argparse
import glob
import os
import os.path as osp
import shutil
from functools import partial

import mmcv
import numpy as np
from PIL import Image


TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def build_compressed_domain_data(video_cdd_filepath, out_dir):
    """
    加载 video_cdd_filepath 中的内容，并将每一帧的内容分开保存到 out_dir 中
    """
    video_cdd_filepath = osp.abspath(video_cdd_filepath)
    assert osp.exists(video_cdd_filepath)
    out_dir = osp.abspath(out_dir)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    video_cdd_dir = osp.dirname(video_cdd_filepath)
    video_cdd_name = osp.basename(video_cdd_filepath)
    video_cdd_name = video_cdd_name.replace('.txt', '')

    # 读取 video_cdd_filepath 中的内容
    cdd = np.loadtxt(video_cdd_filepath, dtype=int, delimiter=' ')

    # 将每一帧的内容分开保存到 out_dir 中
    num_frames = cdd[:, -2].max() + 1
    for frame_id in range(num_frames):
        frame_cdd = cdd[cdd[:, -2] == frame_id]
        frame_cdd_filepath = osp.join(out_dir, f'{video_cdd_name}_frame{frame_id:06d}.txt')
        np.savetxt(frame_cdd_filepath, frame_cdd, fmt='%d')
    

def split_filename(filename):
    name = osp.basename(filename)
    video_name = name.split("frame")[0][:-1]
    idx = int(name.split(".")[0].split("frame")[-1].replace("_", ""))
    return video_name, idx


def build_images_folder(video_frames_folder, out_dir, video_name=""):
    """
    将 video_frames_folder 中的图片保存到 out_dir 中
    可用于VOC数据集的JPEGImages文件夹和SegmentationClass文件夹
    """
    video_frames_folder = osp.abspath(video_frames_folder)
    assert osp.exists(video_frames_folder)
    if video_name == "":
        video_name = osp.basename(video_frames_folder)
    out_dir = osp.abspath(out_dir)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # 读取 video_frames_folder 中的图片
    frame_files = os.listdir(video_frames_folder)
    frame_files = [osp.join(video_frames_folder, frame_file) for frame_file in frame_files]
    frame_files = sorted(frame_files)

    # 将图片保存到 out_dir 中
    for frame_file in frame_files:
        frame_name = osp.basename(frame_file)
        _, frame_idx = split_filename(frame_name)
        frame_suffix = frame_name.split(".")[-1]
        frame_name = osp.join(out_dir, f"{video_name}_frame{frame_idx:06d}.{frame_suffix}")
        shutil.copy(frame_file, frame_name)


def create_folder_not_exists(folder_path):
    """
    如果文件夹不存在, 则创建文件夹
    """
    if not osp.exists(folder_path):
        os.makedirs(folder_path)


def split_dataset(dataset_dir, train_ratio, val_ratio):
    """
    将 CompressedDomainData, JPEGImages, SegmentationClass 三个文件夹中的数据按照比例划分
    """
    dataset_dir = osp.abspath(dataset_dir)
    cdd_dir = osp.join(dataset_dir, 'VOCdevkit', 'VOC2012', 'CompressedDomainData')
    images_dir = osp.join(dataset_dir, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    seg_dir = osp.join(dataset_dir, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
    split_dir = osp.join(dataset_dir, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation')
    create_folder_not_exists(split_dir)

    # 创建 train.txt, val.txt
    with open(osp.join(split_dir, 'train.txt'), 'w') as f_train, open(osp.join(split_dir, 'val.txt'), 'w') as f_val:
        cdd_files = os.listdir(cdd_dir)
        cdd_files = [osp.join(cdd_dir, cdd_file) for cdd_file in sorted(cdd_files)]

        images_files = os.listdir(images_dir)
        images_files = [osp.join(images_dir, images_file) for images_file in sorted(images_files)]

        seg_files = os.listdir(seg_dir)
        seg_files = [osp.join(seg_dir, seg_file) for seg_file in sorted(seg_files)]

        assert len(cdd_files) == len(images_files) == len(seg_files), \
            f'len(cdd_files)={len(cdd_files)}, len(images_files)={len(images_files)}, len(seg_files)={len(seg_files)}'
        for cdd_file, image_file, seg_file in zip(cdd_files, images_files, seg_files):
            cdd_name = osp.basename(cdd_file)
            image_name = osp.basename(image_file)
            seg_name = osp.basename(seg_file)

            cdd_basename = cdd_name.split(".")[0]
            image_basename = image_name.split(".")[0]
            seg_basename = seg_name.split(".")[0]
            assert cdd_basename == image_basename == seg_basename, \
                f'cdd_name={cdd_basename}, image_name={image_basename}, seg_name={seg_basename}'

            if np.random.rand() < train_ratio:
                f_train.write(f'{cdd_basename}\n')
            elif np.random.rand() < train_ratio + val_ratio:
                f_val.write(f'{cdd_basename}\n')


def main():
    parser = argparse.ArgumentParser(description='Build compressed domain data.')
    parser.add_argument('--self_made_dataset_dir', required=True, type=str, help='self-made dataset dir')
    parser.add_argument('--out_dir', type=str, required=True, help='out_dir')
    parser.add_argument('--train_ratio', type=float, default=TRAIN_RATIO, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=VAL_RATIO, help='val ratio')
    args = parser.parse_args()

    self_made_dataset_dir = osp.abspath(args.self_made_dataset_dir)
    out_dir = osp.abspath(args.out_dir)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    assert osp.exists(self_made_dataset_dir)

    frames_folders = glob.glob(osp.join(self_made_dataset_dir, 'images', '*'))
    mvs_paths = glob.glob(osp.join(self_made_dataset_dir, 'mvs_265', '*.txt'))
    label_folders = glob.glob(osp.join(self_made_dataset_dir, 'label', '*', 'pseudo_color_prediction'))

    assert len(frames_folders) == len(mvs_paths) == len(label_folders), \
        f'len(frames_folders)={len(frames_folders)}, len(mvs_paths)={len(mvs_paths)}, len(label_folders)={len(label_folders)}'

    frames_folders = sorted(frames_folders)
    mvs_paths = sorted(mvs_paths)
    label_folders = sorted(label_folders)

    # 创建 VOCdevkit
    voc_devkit_dir = osp.join(out_dir, 'VOCdevkit')
    create_folder_not_exists(voc_devkit_dir)

    # 创建 VOCdevkit/VOC2012
    voc_devkit_voc2012_dir = osp.join(voc_devkit_dir, 'VOC2012')
    create_folder_not_exists(voc_devkit_voc2012_dir)
    
    # 创建 VOCdevkit/VOC2012/ImageSets
    voc_devkit_voc2012_imagesets_dir = osp.join(voc_devkit_voc2012_dir, 'ImageSets')
    create_folder_not_exists(voc_devkit_voc2012_imagesets_dir)
    
    # 创建 VOCdevkit/VOC2012/JPEGImages
    voc_devkit_voc2012_jpegimages_dir = osp.join(voc_devkit_voc2012_dir, 'JPEGImages')
    create_folder_not_exists(voc_devkit_voc2012_jpegimages_dir)

    # 创建 VOCdevkit/VOC2012/SegmentationClass
    voc_devkit_voc2012_segmentationclass_dir = osp.join(voc_devkit_voc2012_dir, 'SegmentationClass')
    create_folder_not_exists(voc_devkit_voc2012_segmentationclass_dir)

    # 创建 VOCdevkit/VOC2012/CompressedDomainData
    voc_devkit_voc2012_compresseddomaindata_dir = osp.join(voc_devkit_voc2012_dir, 'CompressedDomainData')
    create_folder_not_exists(voc_devkit_voc2012_compresseddomaindata_dir)

    for frames_folder, mvs_path, label_folder in zip(frames_folders, mvs_paths, label_folders):
        assert osp.basename(frames_folder) == osp.basename(osp.abspath(osp.join(label_folder, '..'))), \
            f"{frames_folder} != {label_folder}"

        build_compressed_domain_data(mvs_path, voc_devkit_voc2012_compresseddomaindata_dir)
        build_images_folder(frames_folder, voc_devkit_voc2012_jpegimages_dir)
        build_images_folder(label_folder, voc_devkit_voc2012_segmentationclass_dir, 
            video_name=osp.basename(osp.abspath(osp.join(label_folder, '..'))))

        assert len(os.listdir(voc_devkit_voc2012_jpegimages_dir)) == len(os.listdir(voc_devkit_voc2012_compresseddomaindata_dir)), \
            f'len(voc_devkit_voc2012_jpegimages_dir)={len(os.listdir(voc_devkit_voc2012_jpegimages_dir))}, \
                len(voc_devkit_voc2012_compresseddomaindata_dir)={len(os.listdir(voc_devkit_voc2012_compresseddomaindata_dir))}'

        print(f"{frames_folder} done")
    
    split_dataset(out_dir, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()