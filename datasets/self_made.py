from collections import namedtuple
import os
import os.path as osp

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


COLORS = [
    [220, 236, 201], 
    [179, 221, 204], 
    [138, 205, 206], 
    [98, 190, 210],
    [70, 170, 206], 
    [61, 145, 190], 
    [53, 119, 174],
    [45, 94, 158], 
    [36, 68, 142]
]
COLORS = np.array(COLORS)
COLORS = COLORS[:, [2, 1, 0]]


def load_cdd(cdd_filepath, image_width, image_height):
    frame = np.loadtxt(cdd_filepath, delimiter=' ', dtype=int)
    width_max = np.max(frame[:, 2] + frame[:, 4])
    height_max = np.max(frame[:, 3] + frame[:, 5])
    image = np.zeros((height_max, width_max, 3), np.uint8)
    for line in frame:
        mv_y = line[0]
        mv_x = line[1]
        local_y = line[2]
        local_x = line[3]
        mb_height = line[4]
        mb_width = line[5]

        mv_len = np.sqrt(mv_y ** 2 + mv_x ** 2)
        mv_angle = np.arctan2(mv_x, mv_y)
        mv_angle = mv_angle * 180 / np.pi
        mb = int(np.log2(mb_width * mb_height))

        vec = np.array([mv_len, mv_angle, mb])
        mat = np.expand_dims(vec, axis=0)
        mat = np.repeat(mat, mb_height * mb_width, axis=0)
        mat = mat.reshape(mb_width, mb_height, 3)

        image[local_x: local_x+mb_width, local_y: local_y+mb_height] = mat
    
    # 区分横竖
    if image_height > image_width:
        image = image.transpose(1, 0, 2)
    image = image[0:image_height, 0:image_width]

    mv_len = image[..., 0]
    mv_angle = image[..., 1]
    mb = image[..., 2]

    # 着色
    if np.max(mv_len) - np.min(mv_len) > 0:
        image[..., 0] = (mv_len - np.min(mv_len)) / (np.max(mv_len) - np.min(mv_len)) * 255
    image[..., 1] = (mv_angle - -180) / (180 - np.min(mv_angle)) * 255
    image[..., 2] = (mb - 1) / (16 - np.min(mb)) * 255

    # 翻转
    image = np.flip(image, axis=1)
    return image


class SelfMadeGANDataset(data.Dataset):
    """
    使用自己拍摄的数据集来进行训练，分为两个部分：RGB数据和压缩域信息数据。
    RGB数据直接使用图像，压缩域信息使用txt文件
    """
    NUM_CLASSES = 2
    cmap = voc_cmap(N=NUM_CLASSES)
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('road',                 0, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             1, 0, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('building',             2, 0, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 3, 0, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                4, 0, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('pole',                 5, 0, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('traffic light',        6, 0, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         7, 0, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           8, 0, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              9, 0, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  10, 0, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               11, 1, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                12, 0, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  13, 0, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                14, 0, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  15, 0, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('train',                16, 0, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           17, 0, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              18, 0, 'vehicle', 7, True, False, (119, 11, 32)),
    ]
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, 
                 root,
                 split='train',
                 year='2012',
                 rgb_image_suffix='jpg',
                 cdd_suffix='txt',
                 seg_label_suffix='png',
                 transform_rgb=None,
                 transform_cdd=None,
                 rgb_ratio=0.5):
        super().__init__()

        assert split in ['train', 'val'], f'split must be train or val, now is {split}'

        self.root = osp.abspath(root)
        self.year = year
        self.split = split
        self.rgb_image_suffix = rgb_image_suffix
        self.cdd_suffix = cdd_suffix
        self.seg_label_suffix = seg_label_suffix
        self.transform_rgb = transform_rgb
        self.transform_cdd = transform_cdd
        self.rgb_ratio = rgb_ratio

        if self.transform_rgb is None and self.transform_cdd is not None:
            self.transform_rgb = self.transform_cdd
        
        if self.transform_cdd is None and self.transform_rgb is not None:
            self.transform_cdd = self.transform_rgb

        image_dir = osp.join(self.root, 'VOCdevkit', 'VOC' + self.year, 'JPEGImages')
        cdd_dir = osp.join(self.root, 'VOCdevkit', 'VOC' + self.year, 'CompressedDomainData')
        seg_label_dir = osp.join(self.root, 'VOCdevkit', 'VOC' + self.year, 'SegmentationClass')
        
        split_file_path = osp.join(self.root, 'VOCdevkit', 'VOC' + self.year, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file_path, 'r') as f:
            self.image_names = [x.strip() for x in f.readlines()]
        
        self.rgb_image_paths = [osp.join(image_dir, x + '.' + self.rgb_image_suffix) for x in self.image_names]
        self.cdd_paths = [osp.join(cdd_dir, x + '.' + self.cdd_suffix) for x in self.image_names]
        self.seg_label_paths = [osp.join(seg_label_dir, x + '.' + self.seg_label_suffix) for x in self.image_names]

        assert len(self.rgb_image_paths) == len(self.cdd_paths) == len(self.seg_label_paths), \
            f'len(rgb_image_paths)={len(self.rgb_image_paths)}, len(cdd_paths)={len(self.cdd_paths)}, len(seg_label_paths)={len(self.seg_label_paths)}'
    
    def __len__(self):
        return len(self.rgb_image_paths)
    
    def __getitem__(self, idx):
        rgb_image_path = self.rgb_image_paths[idx]
        cdd_path = self.cdd_paths[idx]
        seg_label_path = self.seg_label_paths[idx]

        rgb_image = Image.open(rgb_image_path).convert('RGB')
        seg_label = Image.open(seg_label_path)
        cdd_image = Image.fromarray(load_cdd(cdd_path, rgb_image.width, rgb_image.height))

        if __name__ == "__main__":
            rgb_image.save(f"rgb_{idx}.png")
            cdd_image.save(f"cdd_{idx}.png")
            seg_label.save(f"seg_{idx}.png")

        if rgb_image.height > rgb_image.width:
            rgb_image = rgb_image.rotate(90, expand=True)
            cdd_image = cdd_image.rotate(90, expand=True)
            seg_label = seg_label.rotate(90, expand=True)

        # 选择RGB图像
        if np.random.random() < self.rgb_ratio:
            if self.transform_rgb is not None:
                rgb_image, seg_label = self.transform_rgb(rgb_image, seg_label)
            target = self.encode_target(seg_label)
            return rgb_image, (target, 0)
        else:
            if self.transform_cdd is not None:
                cdd_image, seg_label = self.transform_cdd(cdd_image, seg_label)
            target = self.encode_target(seg_label)
            return cdd_image, (target, 1)
    
    @classmethod
    def encode_target(cls, target):
        """
        将cityscapes的格式转换为0(background)和1(person)
        """
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        return cls.cmap[np.array(target)]


if __name__ == "__main__":
    dataset = SelfMadeGANDataset(
        root="/mnt/tbdisk/bangwhe/experiments/DeepLabV3Plus-Pytorch/datasets/data/self_made_dataset",
        split='train', 
        rgb_ratio=0.5)
    
    gen = iter(dataset)
    _ = next(gen)
