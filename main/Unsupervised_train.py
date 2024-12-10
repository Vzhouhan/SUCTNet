from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import time
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum
from unutils import *
import torch.nn.functional as F
import torchvision.models as models
import numbers
from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
import imgvision as iv
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
                # print(channel_att_sum.shape)
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # print(channel_att_sum.shape,'shape')
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print(scale.shape)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x*scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(module.weight.data,0.0,0.02)


def loss_gradient_difference(real_image,generated): # b x c x h x w
    true_x_shifted_right = real_image[:,:,1:,:]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:,:,:-1,:]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:,:,1:,:]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:,:,:-1,:]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = real_image[:,:,:,1:]
    true_y_shifted_left = real_image[:,:,:,:-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:,:,:,1:]
    generated_y_shift_left = generated[:,:,:,:-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y)**2)/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    igdl = loss_x_gradient + loss_y_gradient
    return igdl


def calculate_x_gradient(images):
    x_gradient_filter = torch.Tensor(
        [
            [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
        ]
    ).cuda()
    x_gradient_filter = x_gradient_filter.view(1, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, x_gradient_filter, groups=1, padding=(1, 1)
    )
    return result


def calculate_y_gradient(images):
    y_gradient_filter = torch.Tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
        ]
    ).cuda()
    y_gradient_filter = y_gradient_filter.view(1, 1, 3, 3)
    result = torch.functional.F.conv2d(
        images, y_gradient_filter, groups=1, padding=(1, 1)
    )
    return result


def loss_igdl( correct_images, generated_images): # taken from https://github.com/Arquestro/ugan-pytorch/blob/master/ops/loss_modules.py
    correct_images_gradient_x = calculate_x_gradient(correct_images)
    generated_images_gradient_x = calculate_x_gradient(generated_images)
    correct_images_gradient_y = calculate_y_gradient(correct_images)
    generated_images_gradient_y = calculate_y_gradient(generated_images)
    pairwise_p_distance = torch.nn.PairwiseDistance(p=1)
    # distances_x_gradient = pairwise_p_distance(
    #     correct_images_gradient_x, generated_images_gradient_x
    # )
    # distances_y_gradient = pairwise_p_distance(
    #     correct_images_gradient_y, generated_images_gradient_y
    # )
    distances_x_gradient = correct_images_gradient_x-generated_images_gradient_x

    distances_y_gradient = correct_images_gradient_y-generated_images_gradient_y

    loss_x_gradient = torch.mean(torch.abs(distances_x_gradient))
    loss_y_gradient = torch.mean(torch.abs(distances_y_gradient))
    loss = 0.5 * (loss_x_gradient + loss_y_gradient)
    return loss

def loss2(correct_images, generated_images):

    # tmp = SSIM_LOSS(img1[:, i:i + 1,:, :], img2[:, i:i + 1,:, :], size=size, sigma=sigma)
    tmp = loss_igdl(correct_images, generated_images)
    loss = tmp.cuda().float()
    return loss.cuda()
##########################################################################
## Layer Norm
class GELU(nn.Module):
    def forward(self, x):
        return F.elu(x)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
class spatialAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(spatialAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # q=to_3d(q)
        # k=to_3d(k)
        # v=to_3d(v)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = ( k.transpose(-2, -1) @ q) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
vgg_conv1_2 = vgg_conv2_2 = vgg_conv3_3 = vgg_conv4_3 = vgg_conv5_3 = None


def conv_1_2_hook(module, input, output):
    global vgg_conv1_2
    vgg_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global vgg_conv2_2
    vgg_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global vgg_conv4_3
    vgg_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global vgg_conv5_3
    vgg_conv5_3 = output
    return None


class CPFE(nn.Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(CPFE, self).__init__()

        self.dil_rates = [2,2, 2]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == 'conv5_3':
            self.in_channels = 512
        elif feature_layer == 'conv4_3':
            self.in_channels = 512
        elif feature_layer == 'conv3_3':
            self.in_channels = 256

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)


        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats
mask_path='mask.mat'
mask3d_batch_test, input_mask_test = init_mask(mask_path, mask_type='Phi', batch_size=10)


class TransformerSODModel(nn.Module):
    def __init__(self):
        super(TransformerSODModel1, self).__init__()

        # Load the [partial] VGG-16 model
        self.vgg16 = models.vgg16(pretrained=True).features

        self.vgg16[0]=nn.Conv2d(28,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        # Extract and register intermediate features of VGG-16
        self.vgg16[3].register_forward_hook(conv_1_2_hook)
        self.vgg16[8].register_forward_hook(conv_2_2_hook)
        self.vgg16[15].register_forward_hook(conv_3_3_hook)
        self.vgg16[22].register_forward_hook(conv_4_3_hook)
        self.vgg16[29].register_forward_hook(conv_5_3_hook)

        # Initialize layers for high level (hl) feature (conv3_3, conv4_3, conv5_3) processing
        self.cpfe_conv3_3 = CPFE(feature_layer='conv3_3')
        self.cpfe_conv4_3 = CPFE(feature_layer='conv4_3')
        self.cpfe_conv5_3 = CPFE(feature_layer='conv5_3')

        # self.cha_att = ChannelwiseAttention(in_channels=384)  # in_channels = 3 x (32 x 4)
        self.cha_att = TransformerBlock(dim=384, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')  # in_channels = 3 x (32 x 4)
        self.hl_conv1 = nn.Conv2d(384, 64, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(64)

        # Initialize layers for low level (ll) feature (conv1_2 and conv2_2) processing
        self.ll_conv_1 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.ll_bn_1 = nn.BatchNorm2d(64)
        self.ll_conv_2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(64)
        self.ll_conv_3 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_3 = nn.BatchNorm2d(64)
        # self.spa_att = SpatialAttention(in_channels=64)
        self.spa_att = TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')  # in_channels = 3 x (32 x 4)
        # Initialize layers for fused features (ff) processing
        self.ff_conv_1 = nn.Conv2d(128, 28, (3, 3), padding=1)
        self.sig=nn.Sigmoid()
    def forward(self, input_):
        global vgg_conv1_2, vgg_conv2_2, vgg_conv3_3, vgg_conv4_3, vgg_conv5_3

        # Pass input_ through vgg16 to generate intermediate features
        self.vgg16(input_)
        # print(vgg_conv1_2.size())
        # print(vgg_conv2_2.size())
        # print(vgg_conv3_3.size())
        # print(vgg_conv4_3.size())
        # print(vgg_conv5_3.size())

        # Process high level features
        conv3_cpfe_feats = self.cpfe_conv3_3(vgg_conv3_3)
        conv4_cpfe_feats = self.cpfe_conv4_3(vgg_conv4_3)
        conv5_cpfe_feats = self.cpfe_conv5_3(vgg_conv5_3)
        # print(conv4_cpfe_feats.shape, conv5_cpfe_feats.shape, 'oko')  # 1,128,64,64
        conv4_cpfe_feats = F.interpolate(conv4_cpfe_feats, scale_factor=2, mode='bilinear', align_corners=True)
        conv5_cpfe_feats = F.interpolate(conv5_cpfe_feats, scale_factor=4, mode='bilinear', align_corners=True)

        # conv4_cpfe_feats = self.up1(conv4_cpfe_feats)
        # conv5_cpfe_feats = self.up2(conv5_cpfe_feats)
        # print(conv4_cpfe_feats.shape, conv5_cpfe_feats.shape, 'oko')#1,128,64,64
        conv_345_feats = torch.cat((conv3_cpfe_feats, conv4_cpfe_feats, conv5_cpfe_feats), dim=1)

        # conv_345_ca, ca_act_reg = self.cha_att(conv_345_feats)
        conv_345_ca = self.cha_att(conv_345_feats)

        conv_345_feats = self.hl_conv1(conv_345_ca)
        conv_345_feats = F.relu(self.hl_bn1(conv_345_feats))
        # print(conv_345_feats.shape, 'jiji')  # 1, 64, 256, 256
        conv_345_feats = F.interpolate(conv_345_feats, scale_factor=4, mode='bilinear', align_corners=True)

        # conv_345_feats = self.up3(conv_345_feats)
        # print(conv_345_feats.shape,'jiji')#1, 64, 256, 256
        # Process low level features
        conv1_feats = self.ll_conv_1(vgg_conv1_2)
        conv1_feats = F.relu(self.ll_bn_1(conv1_feats))
        conv2_feats = self.ll_conv_2(vgg_conv2_2)
        conv2_feats = F.relu(self.ll_bn_2(conv2_feats))
        # print(conv2_feats.shape, 'jijiji')  # 1, 64, 256, 256
        conv2_feats = F.interpolate(conv2_feats, scale_factor=2, mode='bilinear', align_corners=True)

        # conv2_feats = self.up4(conv2_feats )
        # print(conv2_feats.shape,'jijiji')#1, 64, 256, 256
        conv_12_feats = torch.cat((conv1_feats, conv2_feats), dim=1)
        conv_12_feats = self.ll_conv_3(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_3(conv_12_feats))

        conv_12_sa = self.spa_att(conv_12_feats)


        # Fused features
        fused_feats = torch.cat((conv_12_sa, conv_345_feats), dim=1)
        fused_feats = self.sig(self.ff_conv_1(fused_feats)+input_)

        return fused_feats
class KNModel(nn.Module):
    def __init__(self):
        super(KNModel, self).__init__()

        # Load the [partial] VGG-16 model
        self.vgg16 = models.vgg16(pretrained=True).features

        self.vgg16[0]=nn.Conv2d(28,64,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        # Extract and register intermediate features of VGG-16
        self.vgg16[3].register_forward_hook(conv_1_2_hook)
        self.vgg16[8].register_forward_hook(conv_2_2_hook)
        self.vgg16[15].register_forward_hook(conv_3_3_hook)
        self.vgg16[22].register_forward_hook(conv_4_3_hook)
        self.vgg16[29].register_forward_hook(conv_5_3_hook)

        # Initialize layers for high level (hl) feature (conv3_3, conv4_3, conv5_3) processing
        self.cpfe_conv3_3 = CPFE(feature_layer='conv3_3')
        self.cpfe_conv4_3 = CPFE(feature_layer='conv4_3')
        self.cpfe_conv5_3 = CPFE(feature_layer='conv5_3')

        self.cha_att = ChannelwiseAttention(in_channels=384)  # in_channels = 3 x (32 x 4)
        # self.cha_att = TransformerBlock(dim=384, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')  # in_channels = 3 x (32 x 4)
        self.hl_conv1 = nn.Conv2d(384, 64, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(64)

        # Initialize layers for low level (ll) feature (conv1_2 and conv2_2) processing
        self.ll_conv_1 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.ll_bn_1 = nn.BatchNorm2d(64)
        self.ll_conv_2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(64)
        self.ll_conv_3 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_3 = nn.BatchNorm2d(64)

        # self.spa_att = SpatialAttention(in_channels=64)
        self.chan_attn1= TransformerBlock(dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        self.chan_att2= TransformerBlock(dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        self.chan_att3= TransformerBlock(dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        self.spa_attn1 =  TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        self.spa_att2 =  TransformerBlock(dim=64, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        # self.spa_att3 = ChannelwiseAttention(in_channels=32)
        # Initialize layers for fused features (ff) processing
        self.ff_conv_1 = nn.Conv2d(128, 28, (3, 3), padding=1)
        self.sig=nn.Sigmoid()
    def forward(self, input_):
        global vgg_conv1_2, vgg_conv2_2, vgg_conv3_3, vgg_conv4_3, vgg_conv5_3

        # Pass input_ through vgg16 to generate intermediate features
        self.vgg16(input_)
        # print(vgg_conv1_2.size())
        # print(vgg_conv2_2.size())
        # print(vgg_conv3_3.size())
        # print(vgg_conv4_3.size())
        # print(vgg_conv5_3.size())

        # Process high level features
        conv3_cpfe_feats = self.chan_attn1(self.cpfe_conv3_3(vgg_conv3_3))
        # print(conv3_cpfe_feats.shape,'kokik')
        conv4_cpfe_feats = self.chan_att2(self.cpfe_conv4_3(vgg_conv4_3))
        conv5_cpfe_feats = self.chan_att3(self.cpfe_conv5_3(vgg_conv5_3))

        conv4_cpfe_feats = F.interpolate(conv4_cpfe_feats, scale_factor=2, mode='bilinear', align_corners=True)
        conv5_cpfe_feats = F.interpolate(conv5_cpfe_feats, scale_factor=4, mode='bilinear', align_corners=True)

        conv_345_feats = torch.cat((conv3_cpfe_feats, conv4_cpfe_feats, conv5_cpfe_feats), dim=1)

        # conv_345_ca, ca_act_reg = self.cha_att(conv_345_feats)
        # # conv_345_ca = self.cha_att(conv_345_feats)
        # conv_345_feats = torch.mul(conv_345_feats, conv_345_ca)

        conv_345_feats = self.hl_conv1(conv_345_feats)
        conv_345_feats = F.relu(self.hl_bn1(conv_345_feats))
        conv_345_feats = F.interpolate(conv_345_feats, scale_factor=4, mode='bilinear', align_corners=True)

        # Process low level features
        conv1_feats = self.ll_conv_1(vgg_conv1_2)
        # print(conv1_feats.shape,'d,')
        conv1_feats = self.spa_attn1(F.relu(self.ll_bn_1(conv1_feats)))
        conv2_feats = self.ll_conv_2(vgg_conv2_2)
        conv2_feats = self.spa_att2(F.relu(self.ll_bn_2(conv2_feats)))

        conv2_feats = F.interpolate(conv2_feats, scale_factor=2, mode='bilinear', align_corners=True)
        conv_12_feats = torch.cat((conv1_feats, conv2_feats), dim=1)
        conv_12_feats = self.ll_conv_3(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_3(conv_12_feats))

        # Fused features
        fused_feats = torch.cat((conv_12_feats, conv_345_feats), dim=1)
        fused_feats = self.sig(self.ff_conv_1(fused_feats)+input_)

        return fused_feats

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)

    g = torch.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / torch.sum(g)
def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = fspecial_gauss(size, sigma) # window shape [size, size]
    ( channel, _, _) = img1.size()
    # print(window.shape,'window')
    window=window.permute(2,3,0,1).cuda()
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = F.conv2d(img1, window, padding=size // 2, groups=channel).cuda()
    mu2 = F.conv2d(img2, window, padding=size // 2, groups=channel).cuda()
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=size // 2, groups=channel) - mu1_sq.cuda()
    sigma2_sq = F.conv2d(img2*img2, window,padding=size // 2, groups=channel) - mu2_sq.cuda()
    sigma12 = F.conv2d(img1*img2, window,padding=size // 2, groups=channel) - mu1_mu2.cuda()
    v1 = 2*mu1_mu2+C1
    v2 = mu1_sq+mu2_sq+C1
    value = (v1*(2.0*sigma12 + C2))/(v2*(sigma1_sq + sigma2_sq + C2))
    value = torch.mean(value)
    value = 1.0-value
    return value
def ssim_loss(img1, img2, size=11, sigma=1.5):
    loss_list=[]
    for i in range(28):
        # tmp = SSIM_LOSS(img1[:, i:i + 1,:, :], img2[:, i:i + 1,:, :], size=size, sigma=sigma)
        tmp = SSIM_LOSS(img1, img2, size=size, sigma=sigma)
        loss_list.append(tmp.detach().cpu())
    loss = np.mean(loss_list)
    loss=torch.tensor(loss).cuda().float()
    return loss.cuda()
class ZM_bn(nn.Module):
    def __init__(self):
        super(ZM_bn,self).__init__()
    def forward(self,x):
        return x - nn.AdaptiveAvgPool2d(1)(x)

class SAM_Loss(nn.Module):
    def __init__(self):
        super(SAM_Loss, self).__init__()

    def forward(self, output, label):
        ratio = (torch.sum((output + 1e-8).multiply(label + 1e-8), axis=1)) / (torch.sqrt(
            torch.sum((output + 1e-8).multiply(output + 1e-8), axis=1) * torch.sum(
                (label + 1e-8).multiply(label + 1e-8), axis=1)))
        angle = torch.acos(ratio.clip(-1, 1))*180/torch.pi

        return torch.mean(angle)
num=10
sam_val=SAM_Loss()
predimg=np.empty([num,256,256,28])
psnr_list,ssim_list,sam_list=[],[],[]

base = sio.loadmat('Test_2_35.88_0.957_7.21.mat')
#########Contains supervised network results and Truth. The purpose of Truth is to calculate metrics and does not participate in training. 
#########It can be replaced with your HSI to be refined. Due to file size limitations, the supervised network results in the article were not 
#########uploaded and can be sent via private message zhouh@buaa.edu.cn seek.
def init(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform(m.weight)
    elif type(m)==nn.Conv2d:
        nn.init.kaiming_uniform(m.weight)
    elif type(m)==nn.ConvTranspose2d:
        nn.init.kaiming_normal(m.weight)

for i in range(10):

    GT = base['truth'][i]
    GT_t = torch.tensor(GT).float().cuda()
    init =base['pred'][i]
    print('Original:')
    init_to_psnr=torch.tensor(init).T.float().cuda()
    # print(init_to_psnr.shape,'init')
    original_psnr=torch_psnr(init_to_psnr,GT_t.T)
    original_SSIM=torch_ssim(init_to_psnr,GT_t.T)
    original_SAM=sam_val(init_to_psnr,GT_t.T)
    print(original_psnr,'original_PSNR')
    print(original_SSIM,'original——SSIM')
    print(original_SAM,'original_SAM')

    init = torch.tensor(init.T).float().unsqueeze(0).cuda()

    elu=nn.ELU()
    max_epochs =3500
    band = 28
    Net=TransformerSODModel().cuda()
    model = Net
    trainer = torch.optim.AdamW(params=model.parameters(),lr=1e-3)
    schem = torch.optim.lr_scheduler.StepLR(trainer,500,0.95)
    l = nn.L1Loss()
    def TV_Loss(data):
        return torch.abs(data[:,:,1:]-data[:,:,:-1]).mean()+torch.abs(data[:,:,:,1:]-data[:,:,:,:-1]).mean()
    True_m =gen_meas_torch(GT_t.T.unsqueeze(0),mask3d_batch=mask3d_batch_test[i],Y2H=False,mul_mask=False).cuda()
    measuremet1=gen_meas_torch(GT_t.T.unsqueeze(0),mask3d_batch=mask3d_batch_test[i],Y2H=True,mul_mask=False).cuda()
    init_input=torch.cat([init,measuremet1],dim=1)
    better=36.00
    start = time.time()
    class TV_loss():
        def __init__(self):
            return None
        def __call__(self, data):
            return torch.abs(data[:,1:]-data[:,:-1]).mean()
    l2 =  TV_loss()
    for epoch in range(max_epochs):
        trainer.zero_grad()
        pre = model(init)
        # pre+=init
        measurement = gen_meas_torch(pre,mask3d_batch=mask3d_batch_test[i],Y2H=False,mul_mask=False)

        loss = l(measurement, True_m) + 5e-2*ssim_loss(True_m,measurement,size=11,sigma=1.5)+5e-2*loss_igdl(True_m.unsqueeze(0),measurement.unsqueeze(0))
        loss.backward()
        trainer.step()
        schem.step()
        # a = PSNR_GPU(pre,GT_t.T.unsqueeze(0))
        a=torch_psnr(GT_t.T.unsqueeze(0),pre)
        if a>better:
            better=a
            print('\rBest:',epoch,a,end='')
        print('\r','{:.3f}'.format(float(a.detach().cpu().numpy())),'\t','{:.5f}'.format(float(loss.detach().cpu().numpy())),pre.mean(),end='')
        #
    end = time.time()
    print(end-start,'time')
    a = torch_psnr(pre,GT_t.T.unsqueeze(0))
    if a>better:
        better=a
        print(epoch,a)
    print('\r','{:.3f}'.format(float(a.detach().cpu().numpy())),'\t','{:.5f}'.format(float(loss.detach().cpu().numpy())),pre.mean(),end='')
    PSNR, SAM, ERGAS, SSIM, MSE = iv.spectra_metric(pre.permute(0,2,3,1).squeeze(0).detach().cpu().numpy(), GT).get_Evaluation()
    # pre = pre.detach().cpu().numpy()[0].T
    # np.save('0',pre)
    # print()
    # print(pre.shape,'pre')
    # print(GT_t.T.shape,'GT')
    pre=pre.squeeze(0)
    pre_PSNR=torch_psnr(pre,GT_t.T)
    pre_SSIM = torch_ssim(pre, GT_t.T)
    pre_SAM = sam_val(pre, GT_t.T)
    print('pre_PSNR',pre_PSNR)
    print('pre_SSIM',pre_SSIM )
    print('pre_SAM',pre_SAM)
    pre = pre.unsqueeze(0)
    psnr_list.append(pre_PSNR.detach().cpu().numpy())
    ssim_list.append(pre_SSIM.detach().cpu().numpy())
    sam_list.append(SSIM)
    pred = np.transpose(pre.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    print(pred.shape)
    predimg[i]=pred[0]
print(psnr_list,'psnr')
print(ssim_list,'ssim')
print(sam_list,'sam')
psnr_mean = np.mean(np.asarray(psnr_list))
ssim_mean = np.mean(np.asarray(ssim_list))
sam_mean = np.mean(np.asarray(sam_list))
print('psnr',psnr_mean)
print('ssim',ssim_mean)
print('sam',sam_mean)
name =  './' + 'Refine_{}_{:.2f}_{:.3f}_{:.2f}'.format(10,psnr_mean, ssim_mean, sam_mean) + '.mat'
sio.savemat(name,
                 { 'unsuperpred': predimg, 'psnr_list': psnr_mean, 'ssim_list': ssim_mean, 'sam_list': sam_mean})

