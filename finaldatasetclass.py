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
        img_dir = "/media/sandeep/Data/cityscapes/datasets/cityscapes/leftImg8bit/"
        # split = "train"
        ann_dir = "/media/sandeep/Data/cityscapes/datasets/cityscapes/gtFine_trainvaltest/gtFine/"
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "/mnt/hgfs/cityscapes/datasets/cityscapes/leftImg8bit/"))))
        img_dir = os.path.abspath(os.path.join(img_dir, split))
        img_pattern = os.path.join(img_dir, "*", "*_leftImg8bit.png")
        # self.masks = list(sorted(os.listdir(os.path.join(root, "/mnt/hgfs/cityscapes/datasets/cityscapes/gtFine_trainvaltest/gtFine/train"))))

        ann_dir = os.path.abspath(os.path.join(ann_dir, split))

        img_paths = sorted(glob.glob(img_pattern))

        ann_pattern = os.path.join(ann_dir, "*", "*_instanceIds.png")
        ann_paths = sorted(glob.glob(ann_pattern))
        self.img_paths = list(img_paths)
        self.ann_paths = list(ann_paths)
        # print(self.ann_paths)

        self.split = split
        self.CLASSES = ["__background__"]
        self.CLASSES += [l.name for l in csHelpers.labels if l.hasInstances]

        self.initMaps()

        self.cityscapesID_to_ind = {
            l.id: self.name_to_id[l.name] for l in csHelpers.labels if l.hasInstances
        }

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.img_paths[idx]
        mask_path = self.ann_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        ann = Image.open(mask_path)
        # plt.imshow(mask)
        # plt.show()

        ann = np.array(ann)
        ann = torch.from_numpy(ann)

        boxes = []
        labels = []
        masks = []

        self.cityscapesID_to_ind = {
            l.id: self.name_to_id[l.name] for l in csHelpers.labels if l.hasInstances
        }

        instIds = torch.sort(torch.unique(ann))[0]
        print("instIds", instIds)
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            mask = ann == instId

            label = int(instId // 1000)
            label = self.cityscapesID_to_ind[label]

            a = mask.nonzero()
            bbox = [
                torch.min(a[:, 1]),
                torch.min(a[:, 0]),
                torch.max(a[:, 1]),
                torch.max(a[:, 0]),
            ]
            bbox = list(map(int, bbox))

            labels.append(label)
            # masks.append(mask)
            boxes.append(bbox)

        # obj_id = []
        # for instId in instIds:
        #     if instId < 1000:
        #         obj_id.append(instId.numpy())

        numbers = []
        for instid in instIds:
            if instid > 1000:
                numbers.append(instid.numpy())

        masks = ann == numbers[:, None, None]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # masks = torch.cat(masks, dim=0)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # print("box", boxes)
        # print("box shape", boxes.shape)

        # print("labels", labels.shape)

        image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        print("labels shape", labels.shape)
        print("masks shape", masks.shape)
        print("image_id shape", image_id.shape)
        print("boxes shape", boxes.shape)
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        #
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        #
        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_img_info(self, index):
        # Reverse engineered from voc.py
        # All the images have the same size
        return 0

# mask = Image.open('/mnt/hgfs/cityscapes/datasets/cityscapes/gtFine_trainvaltest/gtFine/')


dataset = CityscapesDataset('/', split="train")
dataset[0]