# **Mask RCNN - Cityscapes Dataset**

## What we aim to achieve

To investigate the improvement in accuracy of a Mask-RCNN model trained on Cityscapes dataset with the addition of self-attention layers for the task of instance segmentation.

### Dataset Description

Cityscapes [[Cordts et. al](https://arxiv.org/abs/1604.01685)] is a large-scale dataset and benchmarking tool that consists of images acquired of urban street scenes from a moving vehicle in 50 different cities with dense annotations for pixel-level, instance-level and panoptic labeling tasks. The dataset consists of 30 classes including person,car,bus,road and sky, of which only 10 classes are considered instances or traffic participants. The dataset consists of 5000 images with fine annotations and 20 000 images with course annotations. Of the 5000 images with fine annotations, 2975 images are assigned as train, 500 as validation and the remainder consists of test images with annotations withheld for benchmarking purposes. An example of a train image and its corresponding annotated label is shown in Figure 1.

<img src="/images/frankfurt_000001_027325_leftImg8bit.png" alt="Frankfurt" style="zoom:60%;" />

<img src="/images/frankfurt_000001_027325_gtFine_color.png" alt="Frankfurt_gt" style="zoom:60%;" />

The target information is obtained from polygons in JSON files or InstanceId images that are provided with the dataset. In order to evaluate the and train a Mask-RCNN model with COCO evaluation metrics, the dataset must be loaded in the COCO annotation format for object detection and segmentation. This requires images and targets to be provided in the following format:

**Image**: A PIL images of size (H,W)

**Target**: a dictionary with the following fields:

1. **boxes**:  (FloatTensor[N, 4]): contains the coordinates of N bounding boxes, with [xmin,ymin,xmax,ymax] for every instance in an image.
2. **labels** (Int64Tensor[N]): the label for each bounding box.
3. **masks** (UInt8Tensor[N, H, W]): Segmentation masks for each one of the objects.
4. **image_id** (Int64Tensor[1]): contains a unique id for every image.area (Tensor[N]): The area for each of the bounding boxes.
5. **iscrowd** (UInt8Tensor[N]): specifies whether a segmentation is for an object or groups of objects.

Data-preprocessing must be done to obtain label ids, which must be filtered out for traffic participants. Binary masks must also be produced for every instance in an image. Images with no instances must also be ignored.

 ## Model Description


Mask RCNN is a state of the art deep neural network which solves instance segmentation. There are two main parts of the neural network which are backbone and head, the back bone architecture extracts features from the input images. The backbone for Mask RCNN are ResNet 50, FPN or ResNext 101 [[Kaiming et al.](https://arxiv.org/pdf/1703.06870.pdf)]. The features of the backbone are taken as input in the head , which contains two stages. In the first stage RPN or Region Proposal network scans the output of the backbone layer and it proposes anchor boxes which are bounding boxes with predefined locations and scales relative to images. At the second stage, the neural network scans these region proposed areas and generates object classes, bounding boxes and masks. This stage is called ROI Align. [[LINK](https://medium.com/@alittlepain833/simple-understanding-of-mask-rcnn-134b5b330e95#:~:text=Mask%20RCNN%20is%20a%20deep,two%20stages%20of%20Mask%20RCNN.)]

<img src="/images/Screenshot from 2020-06-17 13-33-49.png" alt="architecture" style="zoom:60%;" />

The mask rcnn backbone currently used is ResNet 50, as it is adaptable to the addition of a self attention layer( which will be explained in later section) .

```
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
```

The pretrained model obtained here is pretrained on the COCO 2017 dataset.When pretrained = True only the last layer of the model will be fine tuned to particular classes, otherwise finetune the whole model. Different backbones can be loaded here. A custom backbone can also be created 

```
in_features = model.roi_heads.box_predictor.cls_score.in_features
```


The region proposal network after getting the anchor boxes tries to tighten the centers of these boxes around the target. This is done through the bounding box regression. After the bounding boxes are obtained, the IOU (intersection over union) with the ground truth values is calculated and classification labels are assigned for such boxes. Box_predictor here is the module that takes the output of bounding boxes and returns the classification labels and distance between the ground truth center and predicted center which is called bounding box regression delta. This distance is then used to calculate the loss values to backpropagate. 

The model takes in the input channels from the in_features and the num_classes which is provided specifically for datasets. For Cityscapes datasets the number of classes = 11 including the background. The FastRCNNPredictor provides the class scores and bounding box regression deltas over the predicted values.# replace the pre-trained head with a new one

```
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

```
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
```

After the bounding boxes and their respective classes are obtained, the masks over the instances of the bounding boxes are calculated. The conv5 mask applies a 2D transposed convolution over the input image.# and replace the mask predictor with a new one

```
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
														hidden_layer,num_classes)
```

This returns the masks for the predicted instances in the image.
Explain loss, accuracy metrics- metrics were getting and what they mean - sandeepDuring training different losses are being calculated such as : classification loss, regression loss for rpn and R-CNN, mask loss.The regression loss in case of rpn and rcnn is calculated using smooth L1 loss, which is regular L1 loss at all the places except at zero. L2 loss is used to smooth the loss at zero. 

```
**psuedocode
if abs(d) < 1/sigma**2
loss = (d*sigma)**2 /2
else
loss = abs(d) — 1/(2*sigma**2)
```

In case of classifications the cross entropy loss is used for both rpn and rcnn.

**Self-Attention**

Self attention [[Ashish](https://arxiv.org/pdf/1706.03762.pdf )] [[Prajit](https://arxiv.org/pdf/1906.05909.pdf)] is a type of attention mechanism that relates different input pixel positions to learn a representation of the input sequence. Given a pixel x<sub>ij</sub>, a memory block is generated which is composed of pixels in positions ab that are in the neighborhood of the pixel x<sub>ij</sub>. The following formula is used to compute the pixel output.

<img src="https://render.githubusercontent.com/render/math?math=y_{ij} = \sum_{a,b Nk(i,j)} softmax_{ab}(q^T_{ij} * k_{ab})*v_{ab}">


Where q<sub>ij</sub>, k<sub>ab</sub> and v<sub>ab</sub> correspond to queries, keys and values respectively. These values are obtained by transformation learned weight matrices W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub>. This computation is done for every pixel value in the memory block. Multiple-attention heads are used where N weight matrices are learned for N groups of pixel features, by dividing the pixel features along the depth dimension. The output of every group or head is then concatenated to produce the final output. Figure x shows an example of the computation performed by a local attention layer.

<img src="/images/Screenshot from 2020-06-17 13-09-00.png" alt="SelfAttention" style="zoom:30%;" />

A local attention layer with kernel size 3. Figure by [Prajit] To encode positional information relative attention is used. The relative distance between pixels in the neighborhood of (i,j) and pixel (i,j) is computed in terms of row and column offsets. An example of relative distance computations is shown in Figure x.

<img src="/images/Screenshot from 2020-06-17 13-09-21.png" alt="Pixels" style="zoom:60%;" />

Relative distance computation in row and column offsets, relative to the highlighted pixel. Figure by [Prajit]

The row and column offsets are associated with embeddings r<sub>a-i</sub> and r<sub>b-j</sub> respectively, These embeddings are concatenated and used to compute the output y<sub>ij</sub>.

<img src="https://render.githubusercontent.com/render/math?math=y_{ij}= \sum_{a,b Nk(i,j)}softmax_{ab(q^T_{ij} * k_{ab} + q^T_{ij} * r_{a-i\,b-j})* v_{ab}" />





The logits used in the computation of the softmax contain information on content and position. The number of parameters in an attention block is independent of the size of the memory block. With convolutions, on the other hand, the parameter count grows quadratically with the size of the kernel.

## Implementation
With the pretrained model from coco dataset 2017, We trained the images for 30 epochs and the predicted mask can be seen below:

<img src="/images/dataset1.png" alt="predicted" style="zoom:10%;" />
