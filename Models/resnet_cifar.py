from __future__ import absolute_import

import torch

'''
This file is from: https://raw.githubusercontent.com/bearpaw/pytorch-classification/master/models/cifar/resnet.py
by Wei Yang
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name should be be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class LightningResnet(pl.LightningModule):
    def __init__(self, depth, num_classes, block_name):
        super().__init__()
        self.model = ResNet(depth, num_classes, block_name)
        self.save_hyperparameters('depth', 'block_name')

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        outputs = self(data)
        pred = torch.argmax(outputs, dim=1)
        loss = F.cross_entropy(outputs, target)
        num_correct = sum([1 if pred[i].item() == target[i].item() else 0 for i in range(len(target))])
        total = len(target)
        batch_dct = {
            'loss' : loss,
            'correct' : num_correct,
            'total' : total
        }
        return batch_dct

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = sum(x["correct"] for x in outputs)
        total = sum([x['total'] for x in outputs])
        self.logger.experiment.add_scalar("Loss/Train", avg_loss.detach(), self.current_epoch)
        self.logger.experiment.add_scalar("Acc/Train", correct/total, self.current_epoch)
        return

    def validation_step(self, batch, batch_idx):
        data, target = batch
        outputs = self(data)
        loss = F.cross_entropy(outputs, target)
        pred = torch.argmax(outputs, dim=1)
        num_correct = sum([1 if pred[i].item() == target[i].item() else 0 for i in range(len(target))])
        total = len(target)
        batch_dct = {
            'val_loss': loss.detach(),
            'correct': num_correct,
            'total': total
        }
        return batch_dct

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum(x["correct"] for x in outputs)
        total = sum([x['total'] for x in outputs])
        self.logger.experiment.add_scalar("Loss/Val", avg_loss.detach(), self.current_epoch)
        self.logger.experiment.add_scalar("Acc/Val", correct/total, self.current_epoch)
        epoch_dct = {
            'val_loss': avg_loss.detach(),
        }
        return epoch_dct

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=.001)
        lr_scheduler = ExponentialLR(optimizer, .9)
        return {'optimizer' : optimizer, 'lr_scheduler' : lr_scheduler}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = sum(x["correct"] for x in outputs)
        total = sum([x['total'] for x in outputs])
        self.logger.experiment.add_scalar("Loss/Test", avg_loss.detach(), self.current_epoch)
        self.logger.experiment.add_scalar("Acc/Test", correct/total, self.current_epoch)

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)