import glob
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
from cityscapesscripts.helpers import csHelpers

from abstract import AbstractDataset


class CityscapesDataset(AbstractDataset):
    def __init__(self, root, split, transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transforms = transforms

        img_dir = "\cityscapes\datasets\cityscapes\leftImg8bit"
        ann_dir = "\cityscapes\datasets\cityscapes\gtFine_trainvaltest\gtFine"

        img_dir = os.path.abspath(os.path.join(img_dir, split))
        img_pattern = os.path.join(img_dir, "*", "*_leftImg8bit.png")

        ann_dir = os.path.abspath(os.path.join(ann_dir, split))
        ann_pattern = os.path.join(ann_dir, "*", "*_instanceIds.png")

        img_paths = sorted(glob.glob(img_pattern))
        ann_paths = sorted(glob.glob(ann_pattern))
        self.img_paths = list(img_paths)
        self.ann_paths = list(ann_paths)

        self.min_area = min_area

        self.split = split
        self.CLASSES = ["__background__"]
        self.CLASSES += [l.name for l in csHelpers.labels if l.hasInstances]

        self.initMaps()

        self.cityscapesID_to_ind = {
            l.id: self.name_to_id[l.name] for l in csHelpers.labels if l.hasInstances
        }

        # filter out images with no instances ##########################
        indices_remove = []
        for ind in range(len(self.ann_paths)):
            ann = torch.from_numpy(np.array(Image.open(self.ann_paths[ind])))
            labels_check = []
            instIds = torch.sort(torch.unique(ann))[0]

            for instId in instIds:

                if int(instId) > 1000:  # group labels
                    label = int(instId // 1000)
                    label = self.cityscapesID_to_ind[label]
                    labels_check.append(label)

            if len(labels_check) == 0:
                indices_remove.append(ind)

        copy_imgs = []
        copy_anns = []

        for x in indices_remove:
            copy_imgs.append(self.img_paths[x])
            copy_anns.append(self.ann_paths[x])

        self.img_paths = [x for x in self.img_paths if x not in copy_imgs]
        self.ann_paths = [x for x in self.ann_paths if x not in copy_anns]

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.img_paths[idx]
        mask_path = self.ann_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        ann = Image.open(mask_path)

        ann_numpy = np.array(ann)  # ann numpy
        ann = torch.from_numpy(ann_numpy)  # ann torch

        labels = []
        boxes = []
        instIds = torch.sort(torch.unique(ann))[0]
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            mask = ann == instId
            label = int(instId // 1000)
            label = self.cityscapesID_to_ind[label]
            labels.append(label)
            a = mask.nonzero()
            bbox = [
                torch.min(a[:, 1]),
                torch.min(a[:, 0]),
                torch.max(a[:, 1]),
                torch.max(a[:, 0]),
            ]
            bbox = list(map(int, bbox))
            boxes.append(bbox)

        area = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            area.append((xmax - xmin) * (ymax - ymin))

        # instances are encoded as different colors
        obj_ids = np.unique(ann_numpy)

        # only keep relevant objects
        obj_ids = np.array([ids for ids in obj_ids if ids >= 1000])

        # split the color-encoded mask into a set
        # of binary masks
        masks = ann_numpy == obj_ids[:, None, None]
        # print("boxes",type(boxes))
        # print("labels",type(labels))
        # print("area",type(area))
        # print("masks",type(masks))
        boxes, masks, labels, area = self._filterGT(boxes, masks, labels, area)
        # print("boxes",type(boxes),boxes)
        # print("labels",type(labels),labels)
        # print("area",type(area),area)
        # print("masks",type(masks),masks)

        # convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx], dtype=torch.int64)

        num_objs = len(labels)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # print("image name",img_path)
        # print("labels shape", labels.shape)
        # print("masks shape", masks.shape)
        # print("image_id shape", image_id.shape)
        # print("boxes shape", boxes.shape)
        # print("area shape", area.shape)
        # print("iscrowd shape", iscrowd.shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

        # filter out instances where the area is less than a certain threshold
        # adapted from mask-rcnn benchmark

    def _filterGT(self, boxes, masks, labels, areas):
        filtered_boxes = []
        filtered_masks = []
        filtered_labels = []
        filtered_area = []
        assert len(masks) == len(labels) == len(boxes) == len(areas)

        for box, mask, label, area in zip(boxes, masks, labels, areas):
            if area < self.min_area:
                continue

            filtered_boxes.append(box)
            filtered_masks.append(mask)
            filtered_labels.append(label)
            filtered_area.append(area)
        mask_default = np.zeros((1024, 2048), dtype=bool)
        if len(filtered_boxes) == 0:
            filtered_boxes = [[0, 0, 10, 50], ]
            filtered_labels = [0, ]
            filtered_masks = [mask_default]
            filtered_area = [500, ]

        return filtered_boxes, filtered_masks, filtered_labels, filtered_area

    def __len__(self):

        return len(self.img_paths)

    def get_img_info(self, index):
        # Reverse engineered from voc.py
        # All the images have the same size
        return 0