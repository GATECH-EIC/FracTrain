""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.autograd as autograd
from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import torch.nn.functional as F


ACT_FW = 0
ACT_BW = 0
GRAD_ACT_ERROR = 0
GRAD_ACT_GC = 0
WEIGHT_BITS = 0
MOMENTUM = 0.9

def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1, pool_size = None, fix_prec=False):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, 
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS, fix_prec=fix_prec)

def conv1x1(in_planes, out_planes, stride=1, pool_size = None, padding=0, fix_prec=False):
    return QConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                   padding=padding, bias=False, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, 
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS, fix_prec=fix_prec)

def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fix_prec=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, fix_prec=fix_prec)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, fix_prec=fix_prec)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits, mask_list):
        residual = x

        out = self.conv1(x, num_bits, num_grad_bits, mask_list)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits, mask_list)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, fix_prec=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, fix_prec=fix_prec)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, fix_prec=fix_prec)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4, fix_prec=fix_prec)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits, mask_list):
        residual = x

        out = self.conv1(x, num_bits, num_grad_bits, mask_list)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits, mask_list)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, num_bits, num_grad_bits, mask_list)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(ResNet, self).__init__()

#         self.num_layers = layers

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_group(block, 64, layers[0], group_id=1)
#         self.layer2 = self._make_group(block, 128, layers[1], group_id=2)
#         self.layer3 = self._make_group(block, 256, layers[2], group_id=3)
#         self.layer4 = self._make_group(block, 512, layers[3], group_id=4)

#         self.avgpool = nn.AvgPool2d(7)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             if isinstance(m, QConv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(0) * m.weight.size(1)
#                 m.weight.data.normal_(0, math.sqrt(2. / n))


#     def _make_group(self, block, planes, layers, group_id):
#         """ Create the whole group"""
#         for i in range(layers):
#             if group_id > 1 and i == 0:
#                 stride = 2
#             else:
#                 stride = 1

#             layer = self._make_layer(block, planes, stride=stride)

#             setattr(self, 'group{}_layer{}'.format(group_id, i), layer)


#     def _make_layer(self, block, planes, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layer = block(self.inplanes, planes, stride, downsample, fix_prec=True)
#         self.inplanes = planes * block.expansion

#         return layer

#     def forward(self, x, num_bits, num_grad_bits):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         for g in range(len(self.num_layers)):
#             for i in range(self.num_layers[g]):
#                 x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits, num_grad_bits)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# def resnet18(pretrained=False, **kwargs):
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model


# def resnet34(pretrained=False, **kwargs):
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     return model


# def resnet50(pretrained=False, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     return model



class RNNGate(nn.Module):
    """Recurrent Gate definition.
    Input is already passed through average pooling and embedding."""
    def __init__(self, input_dim, hidden_dim, proj_dim, rnn_type='lstm'):
        super(RNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        if self.rnn_type == 'lstm':
            self.rnn_one = nn.LSTM(input_dim, hidden_dim)
            # self.rnn_two = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden_one = None
        # self.hidden_two = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, proj_dim)
        # self.proj_two = nn.Linear(hidden_dim, 4)
        self.prob = nn.Sigmoid()
        self.prob_layer = nn.Softmax()

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden_one = repackage_hidden(self.hidden_one)
        # self.hidden_two = repackage_hidden(self.hidden_two)
    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        # self.rnn_one.flatten_parameters()
        # self.rnn_two.flatten_parameters()
        
        out_one, self.hidden_one = self.rnn_one(x.view(1, batch_size, -1), self.hidden_one)
        
        # out_one = F.dropout(out_one, p = 0.1, training=True)
        
        # out_two, self.hidden_two = self.rnn_two(out_one.view(1, batch_size, -1), self.hidden_two)
        
        x_one = self.proj(out_one.squeeze())
        # x_two = self.proj_two(out_two.squeeze())
        
        # proj = self.proj(out.squeeze())
        prob = self.prob_layer(x_one)
        # prob_two = self.prob_layer(x_two)

        # x_one = (prob > 0.5).float().detach() - \
                    # prob.detach() + prob
        
        # x_two = prob_two.detach().cpu().numpy()
        
        x_one = prob.detach().cpu().numpy()
        
        hard = (x_one == x_one.max(axis=1)[:,None]).astype(int)
        hard = torch.from_numpy(hard)
        hard = hard.cuda()
        
        # x_two = hard.float().detach() - \
              # prob_two.detach() + prob_two
            
        x_one = hard.float().detach() - \
                prob.detach() + prob
             
        # print(x_one)

        x_one = x_one.view(x_one.size(0),x_one.size(1), 1, 1, 1)
        
        # x_two = x_two.view(x_two.size(0), x_two.size(1), 1, 1, 1)
        
        return x_one # , x_two



class ResNet_RNN(nn.Module):
    def __init__(self, block, layers, num_classes=1000, embed_dim=40, hidden_dim=20, proj_dim=7):
        self.inplanes = 64
        super(ResNet_RNN, self).__init__()

        self.num_layers = layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.control = RNNGate(embed_dim, hidden_dim, proj_dim, rnn_type='lstm')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.gate_layer1 = nn.Sequential(nn.AvgPool2d(56),
                nn.Conv2d(in_channels=64, out_channels=self.embed_dim, kernel_size=1, stride=1))

        self.layer1 = self._make_group(block, 64, layers[0], group_id=1, pool_size=56)
        self.layer2 = self._make_group(block, 128, layers[1], group_id=2, pool_size=28)
        self.layer3 = self._make_group(block, 256, layers[2], group_id=3, pool_size=14)
        self.layer4 = self._make_group(block, 512, layers[3], group_id=4, pool_size=7)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def _make_group(self, block, planes, layers, group_id, pool_size):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            layer, gate_layer = self._make_layer(block, planes, stride=stride, pool_size=pool_size)

            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)
            setattr(self, 'group{}_gate{}'.format(group_id, i), gate_layer)


    def _make_layer(self, block, planes, pool_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion

        gate_layer = nn.Sequential(
            nn.AvgPool2d(pool_size),
            nn.Conv2d(in_channels=planes * block.expansion,
                      out_channels=self.embed_dim,
                      kernel_size=1,
                      stride=1))

        return layer, gate_layer

    def forward(self, x, bits, grad_bits):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        batch_size = x.size(0)
        self.control.hidden_one = self.control.init_hidden(batch_size)
        
        masks = []

        gate_feature = self.gate_layer1(x)
        mask = self.control(gate_feature)
        
        for g in range(len(self.num_layers)):
            for i in range(self.num_layers[g]):                    

                mask_list = []

                for j in range(len(bits)):
                    mask_list.append(mask[:,j,:,:,:])

                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, bits, grad_bits, mask_list)
                
                mask_list = [mask.squeeze() for mask in mask_list]
                
                masks.append(mask_list)
                    
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask = self.control(gate_feature)
                # mask_grad = self.control_grad(gate_feature)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks


def resnet18_rnn(pretrained=False, **kwargs):
    model = ResNet_RNN(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_rnn(pretrained=False, **kwargs):
    model = ResNet_RNN(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_rnn(pretrained=False, **kwargs):
    model = ResNet_RNN(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model



if __name__ == '__main__':
    model = resnet18()
    from thop import profile
    flops, params = profile(model, inputs=(torch.randn(1, 3, 256, 256),))
    print('flops:', flops, 'params:', params)




