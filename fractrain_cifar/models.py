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

DWS_BITS = 8
DWS_GRAD_BITS = 16
    

def Conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1, fix_prec=False):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, 
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS, fix_prec=fix_prec)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, fix_prec=False):
    return QConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                   padding=padding, dilation=dilation, groups=groups, bias=bias, momentum=MOMENTUM, quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, 
                   quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS, fix_prec=fix_prec)


def make_bn(planes):
	return nn.BatchNorm2d(planes)
	# return RangeBN(planes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fix_prec=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, fix_prec=fix_prec)
        self.bn1 = make_bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, fix_prec=fix_prec)
        self.bn2 = make_bn(planes)
        self.bn3 = make_bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, num_bits, num_grad_bits):
        residual = x

        out = self.conv1(x, num_bits, num_grad_bits)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, num_bits, num_grad_bits)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x, num_bits, num_grad_bits)
            residual = self.bn3(residual)

        out  += residual
        out = self.relu(out)
        return out


########################################
# Original ResNet                      #
########################################

class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16, fix_prec=True)
        self.bn1 = make_bn(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.num_layers = layers
        
        self._make_group(block, 16, layers[0], group_id=1,
                         )
        self._make_group(block, 32, layers[1], group_id=2,
                         )
        self._make_group(block, 64, layers[2], group_id=3,
                         )
        
        # self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
        # for m in self.modules():
            # if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, make_bn):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                
    def _make_group(self, block, planes, layers, group_id=1
                    ):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            layer = self._make_layer_v2(block, planes, stride=stride,
                                       )

            # setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), layer)
            # setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])
            
            
    def _make_layer_v2(self, block, planes, stride=1,
                       ):
        """ create one block and optional a gate module """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, momentum=MOMENTUM,
                    quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS, fix_prec=True)
            
        layer = block(self.inplanes, planes, stride, downsample, fix_prec=True)
        self.inplanes = planes * block.expansion

        # if gate_type == 'ffgate1':
            # gate_layer = FeedforwardGateI(pool_size=pool_size,
                                          # channel=planes*block.expansion)
        # elif gate_type == 'ffgate2':
            # gate_layer = FeedforwardGateII(pool_size=pool_size,
                                           # channel=planes*block.expansion)
        # elif gate_type == 'softgate1':
            # gate_layer = SoftGateI(pool_size=pool_size,
                                   # channel=planes*block.expansion)
        # elif gate_type == 'softgate2':
            # gate_layer = SoftGateII(pool_size=pool_size,
                                    # channel=planes*block.expansion)
        # else:
            # gate_layer = None

        # if downsample:
            # return downsample, layer, gate_layer
        # else:
            # return None, layer, gate_layer
            
        return layer

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, momentum=MOMENTUM, 
                    quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS, fix_prec=True)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fix_prec=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        

    def forward(self, x, num_bits, num_grad_bits):
        x = self.conv1(x, num_bits, num_grad_bits)
        x = self.bn1(x)
        x = self.relu(x)
        
        for g in range(3):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits, num_grad_bits)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


# For CIFAR-10
# ResNet-38

def cifar10_resnet_20(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model

def cifar10_resnet_31(pretrained=False, **kwargs):
    # n = 5
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model



def cifar10_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], **kwargs)
    return model


# ResNet-74
def cifar10_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], **kwargs)
    return model


# ResNet-110
def cifar10_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


# ResNet-152
def cifar10_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], **kwargs)
    return model


# For CIFAR-100
# ResNet-38
def cifar100_resnet_38(pretrained=False, **kwargs):
    # n = 6
    model = ResNet(BasicBlock, [6, 6, 6], num_classes=100)
    return model


# ResNet-74
def cifar100_resnet_74(pretrained=False, **kwargs):
    # n = 12
    model = ResNet(BasicBlock, [12, 12, 12], num_classes=100)
    return model


# ResNet-110
def cifar100_resnet_110(pretrained=False, **kwargs):
    # n = 18
    model = ResNet(BasicBlock, [18, 18, 18], num_classes=100)
    return model


# ResNet-152
def cifar100_resnet_152(pretrained=False, **kwargs):
    # n = 25
    model = ResNet(BasicBlock, [25, 25, 25], num_classes=100)
    return model


# For Recurrent Gate
def repackage_hidden(h):
    """ to reduce memory usage"""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


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
        self.rnn_one.flatten_parameters()
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


class SoftRNNGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='lstm'):
        super(SoftRNNGate, self).__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim)
        else:
            self.rnn = None
        self.hidden = None

        # reduce dim
        self.proj = nn.Linear(hidden_dim, 1)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batch_size,
                                              self.hidden_dim).cuda()))

    def repackage_hidden(self):
        self.hidden = repackage_hidden(self.hidden)

    def forward(self, x):
        # Take the convolution output of each step
        batch_size = x.size(0)
        self.rnn.flatten_parameters()
        out, self.hidden = self.rnn(x.view(1, batch_size, -1), self.hidden)

        proj = self.proj(out.squeeze())
        prob = self.prob(proj)

        x = prob.view(batch_size, 1, 1, 1)
        if not self.training:
            x = (x > 0.5).float()
        return x, prob


class ResNetRecurrentGateSP(nn.Module):
    """SkipNet with Recurrent Gate Model"""
    def __init__(self, block, layers, num_classes=10, gate_dim=32, embed_dim=16, hidden_dim=16, proj_dim=7, gate_type='rnn'):

        self.inplanes = 16

        self.gate_dim = gate_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        super(ResNetRecurrentGateSP, self).__init__()

        self.num_layers = layers
        self.conv1 = conv3x3(3, 16)
        self.bn1 = make_bn(16)
        self.relu = nn.ReLU(inplace=True)

        self.gate_layer1 = nn.Sequential(nn.AvgPool2d(32),
                         nn.Conv2d(in_channels=16, out_channels=self.gate_dim, kernel_size=1, stride=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=self.gate_dim, out_channels=self.embed_dim, kernel_size=1, stride=1))

        self._make_group(block, 16, layers[0], group_id=1, pool_size=32)
        self._make_group(block, 32, layers[1], group_id=2, pool_size=16)
        self._make_group(block, 64, layers[2], group_id=3, pool_size=8)

        # define recurrent gating module
        if gate_type == 'rnn':
            self.control = RNNGate(embed_dim, hidden_dim, proj_dim, rnn_type='lstm')
            # self.control_grad = RNNGate(embed_dim, hidden_dim, proj_dim, rnn_type='lstm')
        elif gate_type == 'soft':
            self.control = SoftRNNGate(embed_dim, hidden_dim, rnn_type='lstm')
            # self.control_grad = RNNGate(embed_dim, hidden_dim, proj_dim, rnn_type='lstm')
        else:
            print('gate type {} not implemented'.format(gate_type))
            self.control = None

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0) * m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_group(self, block, planes, layers, group_id=1, pool_size=16):
        """ Create the whole group"""
        for i in range(layers):
            if group_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_layer_v2(block, planes, stride=stride,
                                       pool_size=pool_size)

            setattr(self, 'group{}_ds{}'.format(group_id, i), meta[0])
            setattr(self, 'group{}_layer{}'.format(group_id, i), meta[1])
            setattr(self, 'group{}_gate{}'.format(group_id, i), meta[2])
            setattr(self, 'group{}_bn{}'.format(group_id, i), meta[3])
            
    def _make_layer_v2(self, block, planes, stride=1, pool_size=16):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:            
            downsample = QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, momentum=MOMENTUM, 
                    quant_act_forward=ACT_FW, quant_act_backward=ACT_BW, quant_grad_act_error=GRAD_ACT_ERROR, quant_grad_act_gc=GRAD_ACT_GC, weight_bits=WEIGHT_BITS)
            
            
        layer = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion
        
        bn = layer.bn3

        gate_layer = nn.Sequential(
            nn.AvgPool2d(int(pool_size)),
            nn.Conv2d(in_channels=planes,
              out_channels=self.gate_dim,
              kernel_size=1,
              stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.gate_dim, out_channels=self.embed_dim, kernel_size=1, stride=1))

        if downsample:
            return downsample, layer, gate_layer, bn
        else:
            return None, layer, gate_layer, None

    def forward(self, x, bits, grad_bits):

        batch_size = x.size(0)
        x = self.conv1(x, 0, 0)
        x = self.bn1(x)
        x = self.relu(x)

        # reinitialize hidden units
        self.control.hidden_one = self.control.init_hidden(batch_size)
        #self.control_grad.hidden_one = self.control_grad.init_hidden(batch_size)
        
        masks = []

        gate_feature = self.gate_layer1(x)
        mask = self.control(gate_feature)
        #mask_grad = self.control_grad(gate_feature)
        
        prev = x

        for g in range(3):
            for i in range(self.num_layers[g]):
                if getattr(self, 'group{}_ds{}'.format(g+1, i)) is not None:
                    prev = getattr(self, 'group{}_ds{}'.format(g+1, i))(prev, 0, 0)
                    prev = getattr(self, 'group{}_bn{}'.format(g+1, i))(prev)
                    
                output_candidates = []
                
                # output_candidates.append(prev)
                
                for k in range(len(bits)):
                    if bits[k] == 0:
                        output_candidates.append(prev)
                    else:
                        out = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, bits[k], grad_bits[k])
                        output_candidates.append(out)
                    
                mask_list = []
                    
                for j in range(len(bits)):
                    mask_list.append(mask[:,j,:,:,:])
                    
                
                prev = x = sum([mask_list[k].expand_as(out) * output_candidates[k] for k in range(len(bits))])
                
                mask_list = [mask.squeeze() for mask in mask_list]
                
                masks.append(mask_list)
                    
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask = self.control(gate_feature)
                # mask_grad = self.control_grad(gate_feature)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, masks


# For CIFAR-10


def cifar10_rnn_gate_20(pretrained=False, **kwargs):
    model = ResNetRecurrentGateSP(BasicBlock, [3, 3, 3], num_classes=10, **kwargs)
    return model

def cifar10_rnn_gate_31(pretrained=False, **kwargs):
    model = ResNetRecurrentGateSP(BasicBlock, [5, 5, 5], num_classes=10, **kwargs)
    return model


def cifar10_rnn_gate_38(pretrained=False, **kwargs):
    """SkipNet-38 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [6, 6, 6], num_classes=10, **kwargs)
    return model


def cifar10_rnn_gate_74(pretrained=False, **kwargs):
    """SkipNet-74 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_classes=10, **kwargs)
    return model


def cifar10_rnn_gate_110(pretrained=False,  **kwargs):
    """SkipNet-110 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=10, **kwargs)
    return model


def cifar10_rnn_gate_152(pretrained=False,  **kwargs):
    """SkipNet-152 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [25, 25, 25], num_classes=10, **kwargs)
    return model


# For CIFAR-100
def cifar100_rnn_gate_38(pretrained=False, **kwargs):
    """SkipNet-38 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [6, 6, 6], num_classes=100, **kwargs)
    return model


def cifar100_rnn_gate_74(pretrained=False, **kwargs):
    """SkipNet-74 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [12, 12, 12], num_classes=100, **kwargs)
    return model


def cifar100_rnn_gate_110(pretrained=False, **kwargs):
    """SkipNet-110 with Recurrent Gate """
    model = ResNetRecurrentGateSP(BasicBlock, [18, 18, 18], num_classes=100, **kwargs)
    return model


def cifar100_rnn_gate_152(pretrained=False, **kwargs):
    """SkipNet-152 with Recurrent Gate"""
    model = ResNetRecurrentGateSP(BasicBlock, [25, 25, 25], num_classes=100, **kwargs)
    return model


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = conv(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = None
        if stride == 1 and in_planes != out_planes:
            self.shortcut = conv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn4 = nn.BatchNorm2d(out_planes)
            

    def forward(self, x, num_bits, num_grad_bits):
        out = F.relu(self.bn1(self.conv1(x, num_bits, num_grad_bits)))
        out = F.relu(self.bn2(self.conv2(out, DWS_BITS, DWS_GRAD_BITS)))
        out = self.bn3(self.conv3(out, num_bits, num_grad_bits))

        if self.stride == 1: 
            if self.shortcut:
                out = out + self.bn4(self.shortcut(x, num_bits, num_grad_bits))
            else:
                out = out + x
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = conv(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self._make_layers(in_planes=32)
        self.conv2 = conv(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.num_layers = [item[2] for item in self.cfg] 

    def _make_layers(self, in_planes):
        
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            strides = [stride] + [1]*(num_blocks-1)

            for j, stride in enumerate(strides):
                setattr(self, 'group{}_layer{}'.format(i+1, j), Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes


    def forward(self, x, num_bits, num_grad_bits):
        x = F.relu(self.bn1(self.conv1(x, num_bits, num_grad_bits)))

        for g in range(7):
            for i in range(self.num_layers[g]):
                x = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, num_bits, num_grad_bits)

        x = F.relu(self.bn2(self.conv2(x, num_bits, num_grad_bits)))

        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def cifar10_mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetV2(num_classes=10, **kwargs)


def cifar100_mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetV2(num_classes=100, **kwargs)




class MobileNetV2_RNN(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, gate_dim=64, embed_dim=32, hidden_dim=32, proj_dim=7):
        super(MobileNetV2_RNN, self).__init__()

        self.num_layers = [item[2] for item in self.cfg]

        self.gate_dim = gate_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = conv(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self._make_layers(in_planes=32)

        self.conv2 = conv(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.control = RNNGate(embed_dim, hidden_dim, proj_dim, rnn_type='lstm')

        self.gate_layer1 = nn.Sequential(nn.AvgPool2d(32),
                         nn.Conv2d(in_channels=32, out_channels=self.gate_dim, kernel_size=1, stride=1),
                         nn.ReLU(),
                         nn.Conv2d(in_channels=self.gate_dim, out_channels=self.embed_dim, kernel_size=1, stride=1))


    def _make_layers(self, in_planes):
        pool_size = 32
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            strides = [stride] + [1]*(num_blocks-1)

            for j, stride in enumerate(strides):
                setattr(self, 'group{}_layer{}'.format(i+1, j), Block(in_planes, out_planes, expansion, stride))

                if stride == 2:
                    pool_size = pool_size/2
                
                gate_layer = nn.Sequential(
                    nn.AvgPool2d(int(pool_size)),
                    nn.Conv2d(in_channels=out_planes,
                      out_channels=self.gate_dim,
                      kernel_size=1,
                      stride=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.gate_dim, out_channels=self.embed_dim, kernel_size=1, stride=1))

                setattr(self, 'group{}_gate{}'.format(i+1, j), gate_layer)

                in_planes = out_planes


    def forward(self, x, bits, grad_bits):
        x = F.relu(self.bn1(self.conv1(x, 0, 0)))

        self.control.hidden_one = self.control.init_hidden(x.size(0))
        
        masks = []

        gate_feature = self.gate_layer1(x)
        mask = self.control(gate_feature)

        for g in range(7):
            for i in range(self.num_layers[g]):                    
                output_candidates = []
                
                for k in range(len(bits)):
                    out = getattr(self, 'group{}_layer{}'.format(g+1, i))(x, bits[k], grad_bits[k])
                    output_candidates.append(out)
                    
                mask_list = []
                    
                for j in range(len(bits)):
                    mask_list.append(mask[:,j,:,:,:])
                    
                x = sum([mask_list[k].expand_as(out) * output_candidates[k] for k in range(len(bits))])
                
                mask_list = [mask.squeeze() for mask in mask_list]
                
                masks.append(mask_list)
                    
                gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
                mask = self.control(gate_feature)

        x = F.relu(self.bn2(self.conv2(x, 0, 0)))

        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x, masks


def cifar10_mobilenet_v2_rnn(pretrained=False, **kwargs):
    return MobileNetV2_RNN(num_classes=10, **kwargs)


def cifar100_mobilenet_v2_rnn(pretrained=False, **kwargs):
    return MobileNetV2_RNN(num_classes=100, **kwargs)






    
