import paddle
import paddle.nn as nn
from paddle.vision.models import resnet50
__all__ = ['ResNetV1b', 'resnet50_v1b', 'resnet50_v1s']



class BasicBlockV1b(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BasicBlockV1b, self).__init__()
        # self.bn_weight = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        self.conv1 = nn.Conv2D(inplanes, planes, 3, stride,
                               dilation, dilation, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2D(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias_attr=False)
        self.bn2 = norm_layer(planes)#, weight_attr = self.bn_weight)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2D):
        super(BottleneckV1b, self).__init__()
        # self.con2d_weight = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal())
        # self.bn_weight = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0))

        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)#, weight_attr=self.con2d_weight)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2D(planes, planes, 3, stride,
                               dilation, dilation, bias_attr=False)#, weight_attr=self.con2d_weight)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)#, weight_attr=self.con2d_weight)
        self.bn3 = norm_layer(planes * self.expansion, bias_attr=None)#, weight_attr=self.bn_weight)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1b(nn.Layer):

    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_stem=False,
                  norm_layer=nn.BatchNorm2D):
        self.inplanes = 128 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        self.con2d_weight = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.KaimingNormal())
        self.bn_weight = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(1))
        self.bn_bias = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.Constant(0))
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2D(3, 64, 3, 2, 1, bias_attr=False, weight_attr=self.con2d_weight),
                norm_layer(64, bias_attr=self.bn_bias, weight_attr=self.bn_weight),
                nn.ReLU(True),
                nn.Conv2D(64, 64, 3, 1, 1, bias_attr=False, weight_attr=self.con2d_weight),
                norm_layer(64, bias_attr=self.bn_bias, weight_attr=self.bn_weight),
                nn.ReLU(True),
                nn.Conv2D(64, 128, 3, 1, 1, bias_attr=False, weight_attr=self.con2d_weight)
            )
        else:
            self.conv1 = nn.Conv2D(3, 64, 7, 2, 3, bias_attr=False, weight_attr=self.con2d_weight)
        self.bn1 = norm_layer(self.inplanes, bias_attr=self.bn_bias, weight_attr=self.bn_weight)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2D(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2D):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion, 1, stride, bias_attr=False, weight_attr=self.con2d_weight),
                norm_layer(planes * block.expansion, bias_attr=self.bn_bias, weight_attr=self.bn_weight)
            )

        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x


def resnet50_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_file = 'res50_paddle.pdparams'
        model.load_dict(paddle.load(model_file))
        print("Pretrained model Resnet50 loaded!")
    # pretrained model from paddle
    #     model_old = resnet50(pretrained=pretrained)
    #     old_dict = model_old.state_dict()
    #     model_dict = model.state_dict()
    #     old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
    #     model_dict.update(old_dict)
    #     model.load_dict(model_dict)
    #     print('pretrained resnet50')
    else:
        print('not pretrained!')
    return model

def resnet50_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)
    model_file = 'resnet50_v1s.pdparams'
    if pretrained:
        model.load_dict(paddle.load(model_file))
        print("Pretrained model Resnet50v1s loaded!")
    else:
        print("not pretrained!")
    return model




if __name__ == '__main__':
    import numpy as np
    x = paddle.randn([1, 3, 224, 224])
    x = paddle.to_tensor(x)
    model = resnet50_v1s(True)
    # for p in model.parameters():
    #     print(p)
