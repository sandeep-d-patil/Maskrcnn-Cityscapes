# -*- coding: utf-8 -*-


from maskrcnn_benchmark.data.datasets import cityscapes
import torch
import numpy as np
import matplotlib.pyplot as plt

from engine import train_one_epoch, evaluate
import utils
import transforms as T

print(cityscapes.__file__)
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = cityscapes.CityScapesDataset("/mnt/hgfs/cityscapes/datasets/cityscapes/leftImg8bit",
                                             "/mnt/hgfs/cityscapes/datasets/cityscapes/gtFine_trainvaltest/gtFine/",
                                             "train", mode="mask")

test_dataset = cityscapes.CityScapesDataset("/mnt/hgfs/cityscapes/datasets/cityscapes/leftImg8bit",
                                            "/mnt/hgfs/cityscapes/datasets/cityscapes/gtFine_trainvaltest/gtFine/",
                                            "test", mode="mask")

len(train_dataset)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# use our dataset and defined transformations

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(train_dataset)).tolist()
train_dataset1 = torch.utils.data.Subset(train_dataset, indices[:-50])
test_dataset1 = torch.utils.data.Subset(test_dataset, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    train_dataset1, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset1, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

train_dataset.CLASSES

type(data_loader.dataset[0][0])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 10

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 1

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)


