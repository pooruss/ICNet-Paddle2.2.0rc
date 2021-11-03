# Description
This repo contains ICNet implemented by Paddlepaddle2.2.0, based on [paper](https://arxiv.org/abs/1704.08545) by Hengshuang Zhao, and et. al(ECCV'18), [code](https://github.com/liminn/ICNet-pytorch) by liminn.
Training and evaluation are done on the [Cityscapes dataset](https://www.cityscapes-dataset.com/) by default.

# Requirements
Python 3.7 or later with the following `pip3 install -r requirements.txt`:
- paddlepaddle-gpu==2.2.0rc0
- numpy==1.17.0
- Pillow==6.0.0
- PyYAML==5.1.2

# Performance  
- Base on Cityscapes dataset, only train on trainning set, and test on validation set, using only one Tesla V100 card on [aistudio platform](https://aistudio.baidu.com/aistudio/index), and input size of the test phase is 2048x1024x3.

|    Method     | Pretrained model |  mIoU(%)  |    GPU     | Time(train) |
| :-----------: | :--------------: | :-------: | :--------: | :---------: |
| ICNet(paper)  |  PSPNet50-half   |   67.7%   |   TitanX   |    80+h     |
| ICNet(paddle) |     [Resnet50-paddle](https://pan.baidu.com/s/1kAvCAghQh01VF32o2EMgTQ)(4hrm)     |   66.7%   | Tesla V100 |     24h     |
| ICNet(paddle) |   [Resnet50-v1s-paddle](https://pan.baidu.com/s/1k7Swsu1QzV4OllKp8IAPDw)(4ug1)   | **69.6%** | Tesla V100 |   **20h**   |

- The evaluating log is in `log/icnet_resnet50_evaluate_log.txt` .

# Demo

|image|predict|
|:---:|:---:|
|![](https://github.com/pooruss/ICNet-paddle/raw/master/demo/frankfurt_000001_020287_leftImg8bit_src.png)|![](https://github.com/pooruss/ICNet-paddle/raw/master/demo/frankfurt_000001_020287_leftImg8bit_mIoU_0.72859335.png)|
|![](https://github.com/pooruss/ICNet-paddle/raw/master/demo/lindau_000056_000019_leftImg8bit_src.png)|![](https://github.com/pooruss/ICNet-paddle/raw/master/demo/lindau_000056_000019_leftImg8bit_mIoU_0.58784664.png) |
|![](https://github.com/pooruss/ICNet-paddle/raw/master/demo/munster_000146_000019_leftImg8bit_src.png)|![](https://github.com/pooruss/ICNet-paddle/raw/master/demo/munster_000146_000019_leftImg8bit_mIoU_0.71236753.png) |
|- All the input images comes from the validation dataset of the Cityscaps, you can switch to the `demo/` directory to check more demo results.||

# Usage
### Preparation
Pretrained models：

- [Resnet50-v1s-mxnet](https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/resnet50_v1s-25a187fa.zip)
- [Resnet50-paddle](https://pan.baidu.com/s/1kAvCAghQh01VF32o2EMgTQ) (transformed from torch): code is 4hrm 
- [Resnet50-v1s-paddle](https://pan.baidu.com/s/1k7Swsu1QzV4OllKp8IAPDw) (transformed from mxnet): code is 4ug1 


### Trainning

First, make sure the pretrained model resnet50_v1s.pdparams exist in the `ICNet-paddle/ `.

Then, modify the configuration in the `configs/icnet.yaml` file:

```Python
### Trainning 
train:
  specific_gpu_num: "1"   # for example: "0", "1" or "0, 1"
  train_batch_size: 7    # adjust according to gpu resources
  cityscapes_root: "./data/Cityscapes/" 
  ckpt_dir: "./ckpt/"     # ckpt and trainning log will be saved here
```
And run : `python train.py`

### Evaluation
First, modify the configuration in the `configs/icnet.yaml` file:
```Python
### Test
test:
  ckpt_path: "./ckpt/icnet_resnet50_126_0.701746612727642_best_model.pdparams"  # set the evaluate model path correctly
```
Then, download the model from:

- [mIoU66.7%-paddle](https://pan.baidu.com/s/1LeS1tktin9Hwu0a3OpkabQ) : code is uqcd
- [mIoU69.6%-paddle](https://pan.baidu.com/s/1eDnEPK5ZYBYLchAOI-6Ovg) : code is 58kq

and put the .pdparams file in  `ckpt/`.

Last : `python evaluate.py`



# Discussion
### Network Structure

The structure of ICNet is mainly composed of `sub4`, `sub2`, `sub1` and `head`: 

- `sub4`: basically a `pspnet`, the biggest difference is a modified `pyramid pooling module`.

- `sub2`: the first three phases convolutional layers of `sub4`, `sub2` and `sub4` share these three phases convolutional layers.

- `sub1`: three consecutive stried convolutional layers, to fastly downsample the original large-size input images

- `head`: through the `CFF` module, the outputs of the three cascaded branches( `sub4`, `sub2` and `sub1`) are connected. Finaly, using 1x1 convolution and interpolation to get the output.

### Issues

- During the training, I found some issues. **Paddlepaddle-2.1.2** does not support constructing optimizers which can specify sublayers' learning rates or other parameters. After updating to **paddlepaddle-2.2.0rc**, the problem is solved. 
- Pretained model Resnet50-v1s is important, and performs better than Resnet50 by 3-4%. Since the <u>Resnet50v1s.pth</u> is not accessible, I transformed the available <u>r50-v1s-pretrained-model.params</u> using mxnet framework to <u>r50v1s-paddle.pdparams</u>. 

### Tricks

Data preprocessing： set the `crop_size` as close as possible to the input size of prediction phase. Here are some experiments based on [liminn-ICNet-pytorch](https://github.com/liminn/ICNet-pytorch) :

- `base_size` to **520**, it means resize the shorter side of image between 520x0.5 and 520x2, and set the `crop size` to **480**, it means randomly crop 480x480 patch to train. The final best mIoU is **66.3%**. ( Resnet50 )
- `base_size` to **1024**, it means resize the shorter side of image between 1024x0.5 and 1024x2, and set the `crop_size` to **960**, it means randomly crop 960x960 patch to train. The final best mIoU is **66.7%**. ( Resnet50 )
- `base_size` to **1024**, it means resize the shorter side of image between 1024x0.5 and 1024x2, and set the `crop_size` to **960**, it means randomly crop 960x960 patch to train. The final best mIoU is **69.6%**. ( Resnet50v1s )
- Beacuse the target dataset is Cityscapes, the image size is 2048x1024, so a large `crop_size`( 960x960 ) is better. It is believed that larger `crop_size` will bring higher mIoU, but large `crop_size` ( such as 1024x1024 ) will result in a smaller batch size and is very time-consuming. 
- set the learning rate of `sub4` to orginal initial learning rate(**0.01**), because it has backbone pretrained weights.
- set the learning rate of `sub1` and `head` to 10 times initial learning rate(**0.1**), because there are no pretrained weights for them.

### Further works

- For experiments in paddle, there are further jobs to do, such as using `crop_size`1024 to see how far can data preprocessing improve the model's performance. 

# Reference
- [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)
- [liminn-ICNet-pytorch](https://github.com/liminn/ICNet-pytorch)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)
- [Human-Segmentation-PyTorch](https://github.com/thuyngch/Human-Segmentation-PyTorch)
