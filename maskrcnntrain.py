import numpy as np
import torch
import torchvision
import torch.utils.data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
from finaldatasetclass import CityscapesDataset
import transforms as T
from PIL import Image

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


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
dataset = CityscapesDataset('/', "train", get_transform(train=True))
dataset_test = CityscapesDataset('/', "val", get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)

indices = np.arange(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[1:20])
dataset_test = torch.utils.data.Subset(dataset_test, indices[1:20])
# print(type(dataset_test))
# dataset = dataset[0]
# dataset_test = dataset_test[0]

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 11

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# let's train it for 10 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)




torch.save(model.state_dict(), "pretrained_true11.pth")

checkpoint = torch.load("pretrained_true11.pth")

model.load_state_dict(checkpoint)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# pick one image from the test set
img, _ = dataset_test[1]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# Plot results:
fig, (ax1, ax2) = plt.subplots(1, 2)

# fig.suptitle('Title of figure', fontsize=20)

# Line plots
# ax1.set_title('Title of ax1')
ax1.imshow(img.mul(255).permute(1, 2, 0).byte().numpy())
# ax1.set_ylim(0,1)


# ax2.set_title('Title of ax2')

predict = torch.squeeze(prediction[0]['masks'])
for i in range(len(predict)):
    predict[i] = predict[i].mul(prediction[0]['scores'][i])

predict_sum = torch.sum(predict, dim=0)

ax2.imshow(predict_sum.mul(255).byte().cpu().numpy())

# plt.tight_layout()
# Make space for title
# plt.subplots_adjust(top=0.85)
plt.show()
plt.savefig('test_1.png')
