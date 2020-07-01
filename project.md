# This is a project.md file.

## Resnet with FPN backbone
Resnet is 
FPN uses a top down architecture with lateral connections to build a feature pyramid from a single scale input as shown in the figure above. The 

https://github.com/mapbox/robosat/issues/60
## Reason where we added attention conv layer, and the values we chose for the same.

Self attention [[Ashish](https://arxiv.org/pdf/1706.03762.pdf )] [[Prajit](https://arxiv.org/pdf/1906.05909.pdf)] is a type of attention mechanism that relates different input pixel positions to learn a representation of the input sequence. Given a pixel x<sub>ij</sub>, a memory block is generated which is composed of pixels in positions ab that are in the neighborhood of the pixel x<sub>ij</sub>. The following formula is used to compute the pixel output.

<img src="https://render.githubusercontent.com/render/math?math=y_{ij} = \sum_{a,b Nk(i,j)} softmax_{ab}(q^T_{ij} * k_{ab})*v_{ab}">


Where q<sub>ij</sub>, k<sub>ab</sub> and v<sub>ab</sub> correspond to queries, keys and values respectively. These values are obtained by transformation learned weight matrices W<sub>Q</sub>, W<sub>K</sub> and W<sub>V</sub>. This computation is done for every pixel value in the memory block. Multiple-attention heads are used where N weight matrices are learned for N groups of pixel features, by dividing the pixel features along the depth dimension. The output of every group or head is then concatenated to produce the final output. Figure x shows an example of the computation performed by a local attention layer.

<img src="/images/Screenshot from 2020-06-17 13-09-00.png" alt="SelfAttention" style="zoom:30%;" />

A local attention layer with kernel size 3. Figure by [Prajit] To encode positional information relative attention is used. The relative distance between pixels in the neighborhood of (i,j) and pixel (i,j) is computed in terms of row and column offsets. An example of relative distance computations is shown in Figure x.

<img src="/images/Screenshot from 2020-06-17 13-09-21.png" alt="Pixels" style="zoom:60%;" />

Relative distance computation in row and column offsets, relative to the highlighted pixel. Figure by [Prajit]

The row and column offsets are associated with embeddings r<sub>a-i</sub> and r<sub>b-j</sub> respectively, These embeddings are concatenated and used to compute the output y<sub>ij</sub>.

<img src="https://render.githubusercontent.com/render/math?math=y_{ij} = \sum_{a,b Nk(i,j)} softmax_{ab}(q^T_{ij} * k_{ab} %20 q^T_{ij} * r_{a-i,b-j})*v_{ab}" />

The logits used in the computation of the softmax contain information on content and position. The number of parameters in an attention block is independent of the size of the memory block. With convolutions, on the other hand, the parameter count grows quadratically with the size of the kernel.

Attention layer can be replaced in two different parts of a Convolution Neural Network. The first part is often referred to as stem layer where the network learns local features of the image such as edges which are used by the later layers to identify global objects. This layer differs from the second part of the Convolution Nueral Network i.e core layer in terms of the input values. The stem layer focuses on simpler operations and large image sizes. While the core layers deal with more complex learnings and smaller patches of images. 

Replacing attention layer with convolution operation at the stem layer poses a challenge, as it underperforms compared to convolution stem of a ResNet [Cite](https://arxiv.org/pdf/1512.03385.pdf) as distance based weights of the convolution layers will learn the edges easily which is required by the higher layers. Also adding the attention layer to Resnet with FPN backbone of the Mask-RCNN requires substantially more gpu computation memory. This posed a challenge as we are using google colab which limits the use of gpu based on availability. Hence the Attention layer is added to the core layer of the ResNet with FPN backbone{Need to explain the different parts of the code properly}. Here the `nn.Conv2d` of first layer of the bottleneck {bottleneck needs to be explained} layer is replaced with the Attention layer. The attention layers were also added to all the bottleneck layers of the backbone in the experiments, but this also increased the gpu computation memory which posed a problem. This problem can be solved in the future by using more gpu memory and also through parallelising the operation with multiple gpu's. 

Attention convolution layer is as shown below.
```AttentionConv
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
```
The attention layer is applied to the first bottleneck layer as shown below:

```BottleneckAttention
self.conv2 = AttentionConv(width, width, kernel_size=7, padding=3, groups=8)
```

Here the inputs <b>width</b> refer to `int(planes * (base_width / 64.)) * groups` which takes in the downscaled input from the conv1x1 layer. The value of width in the first layer of the bottleneck is 64. A lower spatial size of the kernel (kernel_size=3) did not capture the information and improvements are seen by increasing the spatial size. A kernel size of 7 was found to be more appropriate according the paper [[Ashish](https://arxiv.org/pdf/1706.03762.pdf )]. The padding size is chosen to be 3 {Why?} and the groups denote the number of attention heads used. The attention heads refer to the output of one attention layer. The value for groups should be chosen such that the output channels of the attention layer is divisible by the number of groups.

##

|     Model                      |  AP<sup>bb</sup> | AP<sub>50</sub><sup>bb</sup> | AP<sub>75</sub><sup>bb</sup> |  AP<sup>seg</sup> | AP<sub>50</sub><sup>seg</sup> | AP<sub>75</sub><sup>seg</sup> |
|------------------------------------------|-------|------|------|-------|-------|-------|
| Pretrained mask-rcnn (10 epochs)         | 18.9  | 33.0 | 17.1 | 15.0  | 31.5  |  11.4 |
| mask-rcnn from scratch (20 epochs)       | 1.0   | 3.2  |  0.1 |  0.7  |  2.4  |  0.1  |
| mask-rcnn from scratch (10 epochs)       | 1.0   | 3.2  |  0.1 |  0.7  |  2.4  |  0.1  |
| mask-rcnn with self attention (10 epochs)| 0.8   | 2.8  |  0.1 |  0.6  | 1.9   |  0.0  | 



<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Model</th>
    <th class="tg-0lax">Total no. parameters</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">resnet50_fpn mask-rcnn</td>
    <td class="tg-0lax">43,970,833</td>
  </tr>
  <tr>
    <td class="tg-c3ow">resnet50_fpn mask-rcnn with self-attention</td>
    <td class="tg-0lax">43,898,449</td>
  </tr>
</tbody>
</table>

As can be seen from the table above, the number of total parameters are reduced by 72,384. This is in agreement with the findings in [[Prajit](https://arxiv.org/abs/1906.05909)]
