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

        # load all image files, sorting them to
        # ensure that they are aligned

        img_dir = "/home/sandeep/Downloads/cityscapes/leftImg8bit/"
        ann_dir = "/home/sandeep/Downloads/cityscapes/gtFine/"

        img_dir = os.path.abspath(os.path.join(img_dir, split))
        img_pattern = os.path.join(img_dir, "*", "*_leftImg8bit.png")

        ann_dir = os.path.abspath(os.path.join(ann_dir, split))
        ann_pattern = os.path.join(ann_dir, "*", "*_instanceIds.png")

        img_paths = sorted(glob.glob(img_pattern))
        ann_paths = sorted(glob.glob(ann_pattern))
        self.img_paths = list(img_paths)
        self.ann_paths = list(ann_paths)

        self.split = split
        self.CLASSES = ["__background__"]
        self.CLASSES += [l.name for l in csHelpers.labels if l.hasInstances]

        self.initMaps()

        self.cityscapesID_to_ind = {
            l.id: self.name_to_id[l.name] for l in csHelpers.labels if l.hasInstances
        }

    def __getitem__(self, idx):
        # load images and masks
        # idx = 287
        img_path = self.img_paths[idx]
        mask_path = self.ann_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # print("imag_path", img_path)
        # print("idx", idx)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        ann = Image.open(mask_path)

        ann_numpy = np.array(ann)  # ann numpy
        ann = torch.from_numpy(ann_numpy)  # ann torch

        obj_ids = np.unique(ann_numpy)

        # print("obj_ids", obj_ids)
        # only keep relevant objects
        obj_ids = np.array([ids for ids in obj_ids if ids >= 1000])
        # print("obj_id length", len(obj_ids))
        # print("obj_id", obj_ids)

        # if len(obj_ids) == 0:
        #     empty_ann_path = self.get_img_info(idx)["ann_path"]
        #     print("EMPTY ENTRY:", empty_ann_path)
        #     self.img_paths.pop(idx)
        #     self.ann_paths.pop(idx)

        while len(obj_ids) != 0:
            ### create labels
            labels = []
            instIds = torch.sort(torch.unique(ann))[0]
            for instId in instIds:
                if instId < 1000:  # group labels
                    continue

                label = int(instId // 1000)
                label = self.cityscapesID_to_ind[label]
                labels.append(label)

            masks = ann_numpy == obj_ids[:, None, None]
            # print(len(masks))

            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # convert to tensor
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            image_id = torch.tensor([idx])

            # print("labels shape", labels)
            # print("masks shape", masks)
            # print("image_id shape", image_id)
            # print("boxes shape", boxes)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            # print("labels shape", area.shape)
            # print("masks shape", masks.shape)
            # print("image_id shape", image_id.shape)
            # print("boxes shape", boxes.shape)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        # masks = ann_numpy == obj_ids[:, None, None]
        # print("obj_ids", obj_ids)

        boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        # print("boxes shape", boxes.shape)
        labels = torch.tensor([0], dtype=torch.int64)
        masks = torch.zeros((1, 1024, 2048), dtype=torch.uint8)
        # print("masks",masks.shape)
        image_id = torch.tensor([idx])
        area = torch.tensor([[0.0]])
        # num_objs = len(obj_ids)
        iscrowd = torch.zeros(1, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target





    def __len__(self):
        return len(self.img_paths)

    def get_img_info(self, index):

        # Reverse engineered from voc.py
        # All the images have the same size
        return {
            "height": 1024,
            "width": 2048,
            "idx": index,
            "img_path": self.img_paths[index],
            "ann_path": self.ann_paths[index],
        }


# dataset = CityscapesDataset('/', split="val")
# dataset[2]

