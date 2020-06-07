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
    def __init__(self, root, transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to
        # ensure that they are aligned
        img_dir= "/mnt/hgfs/cityscapes/datasets/cityscapes/leftImg8bit/"
        split="train"
        ann_dir = "/mnt/hgfs/cityscapes/datasets/cityscapes/gtFine_trainvaltest/gtFine/"
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
        print(self.ann_paths)

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
        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        plt.imshow(mask)
        plt.show()

        mask = np.array(mask)


        # dataset = CityscapesDataset('/')
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        print("obj_ids",obj_ids)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        print("obj_ids_withoutbackground", obj_ids)
        #
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        #
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        print("box", boxes)
        print("box shape", boxes.shape)
        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels = []
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        self.cityscapesID_to_ind = {
            l.id: self.name_to_id[l.name] for l in csHelpers.labels if l.hasInstances
        }
        instIds = torch.sort(torch.unique(torch.from_numpy(mask)))[0]
        for instId in instIds:
            if instId < 1000:  # group labels
                continue

            # masks = masks == instId
            label = int(instId // 1000)
            label = self.cityscapesID_to_ind[label]
            labels.append(label)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
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
