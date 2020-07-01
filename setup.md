# File setup

## Required libraries are:
```libraries
1. PYCOCOtools
2. Cityscapesscripts
3. Torch >= 1.5
4. torchvision>=0.6.0
5. torchsummary
```

To run the Mask RCNN model without changing the self attention model. Run the following notebook. [click here](https://github.com/sandeeprockstar/Maskrcnn-Cityscapes/blob/master/MaskRCNNresnet_with_FPN.ipynb)
Follow the instructions mentioned in the notebook to train the model with/without pretrained weights.

To run the Mask RCNN model with changing a bottleneck layer 
Copy the Code from ResNet model into the `torchvision.model.resnet.py` which can be found in the downloaded library and run the model.
