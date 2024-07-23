import os
import math
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import einops
    from timm.models.layers import trunc_normal_, DropPath
    from timm.models.registry import register_model
except ImportError:
    print("The 'some_library' module is not installed. Please install it using 'pip install some_library'.")



class SVD_Layer(nn.Module):
    def __init__(self, input_size, output_size, singular_factor=1):
        super(SVD_Layer, self).__init__()
        assert output_size == 4 * input_size or input_size == 4 * output_size, 'Only MLP factor 4 is allowed'
        inputs = list()
        outputs = list()
        size = input_size if output_size > input_size else output_size
        if size == 64:
            inputs = [4, 4, 4]
            outputs = [8, 4, 8]
        elif size == 96:
            inputs = [4, 6, 4]
            outputs = [8, 6, 8]
        elif size == 128:
            inputs = [4, 8, 4]
            outputs = [8, 8, 8]
        elif size == 192:
            inputs = [4, 12, 4]
            outputs = [8, 12, 8]
        elif size == 256:
            inputs = [4, 16, 4]
            outputs = [8, 16, 8]
        elif size == 384:
            inputs = [8, 6, 8]
            outputs = [16, 6, 16]
        elif size == 512:
            inputs = [8, 8, 8]
            outputs = [16, 8, 16]
        elif size == 768:
            inputs = [8, 12, 8]
            outputs = [16, 12, 16]
        else:
            print('Error: input_size is not allowed!')
        if output_size < input_size:
            tmp = inputs
            inputs = outputs
            outputs = tmp

        a, b, c = inputs[2]*outputs[2], inputs[0]*outputs[0] + inputs[1]*outputs[1], -1*input_size*output_size
        num_singularValues = (-1*(b) + math.sqrt(b**2 - 4*a*c))/(2*a)
        num_singularValues = int(num_singularValues // singular_factor)
        self.inputs = inputs
        self.outputs = outputs

        self.W_1 = nn.Parameter(torch.empty((inputs[0], outputs[0], num_singularValues), dtype=torch.float32, requires_grad=True))
        self.W_2 = nn.Parameter(torch.empty((num_singularValues, inputs[1], outputs[1], num_singularValues), dtype=torch.float32, requires_grad=True))
        self.W_3 = nn.Parameter(torch.empty((num_singularValues, inputs[2], outputs[2]), dtype=torch.float32, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.W_1, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.W_2, math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.W_3, math.sqrt(2))
        self.b = torch.nn.Parameter(torch.zeros((output_size,), dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        W = torch.einsum('i o s, s I O S, S a b -> i I a o O b', self.W_1, self.W_2, self.W_3)
        W = einops.rearrange(W, 'a b c d e f -> (a b c) (d e f)')
        y = torch.matmul(x, W) + self.b
        return y


# Works for both 2D and 3D
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, ten_net=0):
        super(TransformerBlock, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = LayerNorm(dim, data_format='channels_last')
        self.attn = Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, data_format='channels_last')
        mlp_hidden_dim = int(dim * 4)
        self.pwconv1 = nn.Linear(dim, 4 * dim) if ten_net == 0 else SVD_Layer(dim, 4 * dim, ten_net)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) if ten_net == 0 else SVD_Layer(4 * dim, dim, ten_net)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        #x = x.flatten(2).transpose(1, 2)
        if self.gamma1 is not None:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        input = x
        #x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = input + self.drop_path(x)
        #x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, ten_net=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) if ten_net==0 else SVD_Layer(dim, 4*dim, ten_net)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) if ten_net == 0 else SVD_Layer(4*dim, dim, ten_net)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class TransformerBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, ten_net=0):
        super(TransformerBlock3D, self).__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = LayerNorm(dim, data_format='channels_last')
        self.attn = Attention(dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim, data_format='channels_last')
        mlp_hidden_dim = int(dim * 4)
        self.pwconv1 = nn.Linear(dim, 4 * dim) if ten_net == 0 else SVD_Layer(dim, 4 * dim, ten_net)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) if ten_net == 0 else SVD_Layer(4 * dim, dim, ten_net)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, metadata=None):
        x = x + self.pos_embed(x)
        B, N, H, W, D = x.shape
        x = einops.rearrange(x, 'b c h w d -> b (h w d) c')

        if metadata != None:
            x = torch.cat((x, metadata.unsqueeze(1)), dim=1)

        if self.gamma1 is not None:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))

        input = x
        #x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = input + self.drop_path(x)

        if metadata != None:
            x, metadata = x[:, :-1, :], x[:, -1, :]

        #x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = einops.rearrange(x, 'b (h w d) c -> b c h w d', h=H, w=W, d=D)
        return x, metadata


class Block3d(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, ten_net=0):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm3d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) if ten_net==0 else SVD_Layer(dim, 4*dim, ten_net) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) if ten_net == 0 else SVD_Layer(4*dim, dim, ten_net)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000,
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6, 
                 head_init_scale=1., 
                 ten_net=0, 
                 use_transformer=False,
                 ):
        
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        num_stages = 3 if use_transformer else 4
        for i in range(num_stages):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, ten_net=ten_net) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        if use_transformer:
            stage = nn.ModuleList(
                [TransformerBlock(dim=dims[-1], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, ten_net=ten_net) for j in range(depths[-1])])
            self.stages.append(stage)
            cur += depths[-1]

        self.use_transformer = use_transformer

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias != None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        num_cnnlayers = len(self.stages) - 1 if self.use_transformer else len(self.stages)
        for i in range(num_cnnlayers):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if self.use_transformer:
            x = self.downsample_layers[-1](x)
            #No metadata is needed for the 2d-model
            for s in self.stages[-1]:
                x = s(x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvNeXt3d(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., ten_net=0, use_transformer=False,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm3d(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm3d(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        num_stages = 3 if use_transformer else 4
        for i in range(num_stages):
            stage = nn.Sequential(*[Block3d(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, ten_net=ten_net) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        if use_transformer:
            stage = nn.ModuleList([TransformerBlock3D(dim=dims[-1], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value, ten_net=ten_net) for j in range(depths[-1])])
            self.stages.append(stage)
            cur += depths[-1]

        self.use_transformer = use_transformer

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.pre_head = nn.Linear(dims[-1], dims[-1])
        self.metadata_prehead = nn.Linear(4, dims[-1], bias=False)
        self.head = nn.Linear(dims[-1], num_classes)

        self.metadata_embedding = nn.Linear(4, dims[-1])

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias != None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x, get_features=False, train_stages=[0, 1, 2, 3]):
        def apply_stage(x, i):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            return x

        feature_list = []
        num_cnnlayers = len(self.stages) - 1 if self.use_transformer else len(self.stages)
        for i in range(num_cnnlayers):
            if i in train_stages:
                x = apply_stage(x, i)
            else:
                with torch.no_grad():
                    x = apply_stage(x, i)
            if get_features: feature_list.append(x)
        if self.use_transformer:
            metadata = None
            x = self.downsample_layers[-1](x)
            if num_cnnlayers in train_stages:
                for s in self.stages[-1]:
                    x, metadata = s(x, metadata)
            else:
                with torch.no_grad():
                    for s in self.stages[-1]:
                        x, metadata = s(x, metadata)
            if get_features: feature_list.append(x)
            
        if get_features:
            return feature_list
        else:
            return self.norm(x.mean([-3, -2, -1]))  # global average pooling, (N, C, D, H, W) -> (N, C)  # global average pooling, (N, C, D, H, W) -> (N, C)

    def forward(self, x, get_features=False, train_stages=[0, 1, 2, 3]):
        x = self.forward_features(x, get_features, train_stages)

        if get_features:
            return x
        else:
            x = self.pre_head(x)
            x = F.relu(x)
            x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm3d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, depth, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, depth, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
}


@register_model
def convnext_tiny(pretrained=False, ten_net=0, in_chan=3, use_transformer=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], ten_net=ten_net, in_chans=in_chan, use_transformer=use_transformer, **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model


""" Relevant architecures after this point for the STOIC Challenge """

def load_params3Dfromparams2D(model2d, model3d, init_mode, init_kind, added_dim=2):
    for p2d, p3d in zip(model2d.named_parameters(), model3d.named_parameters()):
        p2d_name, p2d = p2d
        p3d_name, p3d = p3d
        if not p2d_name.startswith("head") and len(p2d.shape) == 4 and (p2d.shape[-1] == p2d.shape[-2]):
            weight2d = p2d.detach().data
            w3d_size = p3d.size()

            scaled_weight = torch.zeros_like(p3d.data)
            sws = scaled_weight.shape

            mu = sws[-1] // 2
            sigma = mu / 4
            normal = lambda input: exp(-(input - mu) ** 2 / (2 * sigma ** 2))

            if init_mode == 'full':
                scaled_weight = einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d1 d2 d3", d3=weight2d.shape[-1],) / w3d_size[2 + added_dim]
            elif init_mode == 'one':
                if init_kind == 'm' and sws[-1] >= 3:
                    scaled_weight[:, :, :, :, (sws[-1] // 2) - 1:(sws[-1] // 2) + 2] = einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d1 d2 d3", d3=3)
                    scaled_weight /= 3
                elif init_kind == 'g' and sws[-1] >= 3:
                    for d3 in range(sws[-1]):
                        scaled_weight[:, :, :, :, d3] = weight2d * normal(d3)
                        scaled_weight *= weight2d.sum() / scaled_weight.sum()
                else:
                    scaled_weight[:, :, :, :, sws[-1] // 2] = weight2d
            elif init_mode == 'two':
                if init_kind == 'm' and sws[-1] >= 3 and sws[-2] >= 3:
                    scaled_weight[:, :, :, :, (sws[-1] // 2) - 1:(sws[-1] // 2) + 2] += einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d1 d2 d3", d3=3)
                    scaled_weight[:, :, :, (sws[-2] // 2) - 1:(sws[-2] // 2) + 2, :] += einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d1 d3 d2", d3=3)
                    scaled_weight *= weight2d.sum() / scaled_weight.sum()
                elif init_kind == 'g' and sws[-1] >= 3:
                    for d3 in range(sws[-1]):
                        scaled_weight[:, :, :, :, d3] += weight2d * normal(d3)
                    for d2 in range(sws[-2]):
                        scaled_weight[:, :, :, d2, :] += weight2d * normal(d2)
                    scaled_weight *= weight2d.sum() / scaled_weight.sum()
                else:
                    scaled_weight[:, :, :, :, sws[-1] // 2] = weight2d
                    scaled_weight[:, :, :, sws[-2] // 2, :] = weight2d
                    scaled_weight *= weight2d.sum() / scaled_weight.sum()
            else:
                if init_kind == 'm' and sws[-1] >= 3 and sws[-2] >= 3 and sws[-3] >= 3:
                    scaled_weight[:, :, :, :, (sws[-1] // 2) - 1:(sws[-1] // 2) + 2] += einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d1 d2 d3", d3=3)
                    scaled_weight[:, :, :, (sws[-2] // 2) - 1:(sws[-2] // 2) + 2, :] += einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d1 d3 d2", d3=3)
                    scaled_weight[:, :, (sws[-3] // 2) - 1:(sws[-2] // 2) + 2, :, :] += einops.repeat(weight2d, "c1 c2 d1 d2 -> c1 c2 d3 d1 d2", d3=3)
                    scaled_weight *= weight2d.sum() / scaled_weight.sum()
                elif init_kind == 'g' and sws[-1] >= 3:
                    for d3 in range(sws[-1]):
                        scaled_weight[:, :, :, :, d3] += weight2d * normal(d3)
                    for d2 in range(sws[-2]):
                        scaled_weight[:, :, :, d2, :] += weight2d * normal(d2)
                    for d1 in range(sws[-2]):
                        scaled_weight[:, :, d1, :, :] += weight2d * normal(d1)
                    scaled_weight *= weight2d.sum() / scaled_weight.sum()
                else:
                    scaled_weight[:, :, :, :, sws[-1] // 2] = weight2d
                    scaled_weight[:, :, :, sws[-2] // 2, :] = weight2d
                    scaled_weight[:, :, sws[-3] // 2, :, :] = weight2d
                    scaled_weight *= weight2d.sum() / scaled_weight.sum()
            p3d.data = scaled_weight
        elif not p2d_name.startswith("head"):
            p3d.data = p2d.detach().data
    return model3d


def load_params2d(model2d, pretrained_path, ten_net, use_transformer, size='tiny'):

    modelname = 'convnextransformer' if use_transformer else 'convnext'
    tennetname = ''.join(('ten_net', str(ten_net)))
    name = 'pretrained_model.pth'
    path = os.path.join(pretrained_path, modelname, tennetname, name)
    print(path)
    model2d.load_state_dict(torch.load(path)['model'])
    return model2d


def load_params3Dfromparams3D(model, pretrained_path, ten_net, use_transformer, size='small', datasize=256, pretrained_mode='multitaskECCV'):

    modelname = 'convnextransformer' if use_transformer else 'convnext'
    
    tennetname = ''.join(('ten_net', str(ten_net)))
    name = 'pretrained_model.pth' if datasize == 256 else 'pretrained_model128.pth'
    
    if pretrained_mode == 'segmentation':
        path = os.path.join(pretrained_path, 'segmentation', modelname, tennetname, name)
    elif pretrained_mode == 'multitask':
        path = os.path.join(pretrained_path, 'multitask', modelname, tennetname, name)
    elif pretrained_mode == 'trainedStoic' or pretrained_mode == 'stoic':
        path = os.path.join(pretrained_path, 'trainedStoic', modelname, tennetname, name)
    elif pretrained_mode in ['segmia', 'segmentationECCV', 'segmiaECCV', 'multitaskECCV', 'multimiaECCV', 'segmiaECCVFull', 'segmentationECCVFull', 'multitaskECCVFull']:
        path = os.path.join(pretrained_path, pretrained_mode, modelname, tennetname, name)
    elif pretrained_mode in ['trainedMiaInf', 'trainedMiaInfECCV']:
        path = os.path.join(pretrained_path, 'trainedMiaInf', modelname, tennetname, name)
    else:
        name = ''.join(('pretrained_model_', pretrained_mode, '.pth'))
        path = os.path.join(pretrained_path, 'multitask', modelname, tennetname, name)
    checkpoint = torch.load(path)
    model_sd = model.state_dict()

    # Because I fucked up saving the segmentation pretraining models...
    if 'encoder_state_dict' in checkpoint:
        for name in checkpoint['encoder_state_dict']:
            model_sd['.'.join(name.split('.')[1:])] = checkpoint['encoder_state_dict'][name]
    elif 'model_ema_state_dict' in checkpoint:
        counter = 0
        for name in checkpoint['model_ema_state_dict']:
            if name.split('.')[0] == 'encoder':
                counter += 1
                model_sd['.'.join(name.split('.')[2:])] = checkpoint['model_ema_state_dict'][name]
        if counter == 0: print('ATTENTION: Loading of pretrained weigths does not work correctly!!!!')
    elif 'model_state_dict' in checkpoint:
        counter = 0
        for name in checkpoint['model_state_dict']:
            # print(name.split('.')[0])
            if name.split('.')[0] == 'encoder':
                counter += 1
                model_sd['.'.join(name.split('.')[2:])] = checkpoint['model_state_dict'][name]
        if counter == 0: print('ATTENTION: Loading of pretrained weigths does not work correctly!!!!')
    elif pretrained_mode in ['trainedStoic', 'stoic', 'trainedMiaInf', 'trainedMiaInfECCV']:
        for name in checkpoint['model_ema']:
            model_sd['.'.join(name.split('.')[1:])] = checkpoint['model_ema'][name]
    else:
        for name in checkpoint['model_ema']:
            model_sd['.'.join(name.split('.')[1:])] = checkpoint['model_ema_state_dict'][name]
    model.load_state_dict(model_sd)
    return model



def convnext_tiny_3d(pretrained=True, added_dim=2, init_mode='full', ten_net=0, in_chan=3, use_transformer=False, pretrained_path=None, pretrained_mode='imagenet', drop_path=0.1, datasize=256, **kwargs):
    assert added_dim in [2] # [0,1,2] # Symbolic for (D)epth, (H)eight, (W)idth
    pretrained_path="/home/lisadesanti/DeepLearning/Lisa/pretrained_backbones/convnext3d_pretrained/saved_models/"
    model3d = ConvNeXt3d(num_classes=2, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], ten_net=ten_net, in_chans=in_chan, use_transformer=use_transformer, drop_path_rate=drop_path, **kwargs)

    init_mode, init_kind = (init_mode, None) if init_mode[-2] != '_' else init_mode.split('_')
    print('pretrained=', pretrained)
    print('pretrained_mode=', pretrained_mode)
    print('init_mode=', init_mode)
    print('init_kind=', init_kind)
    if pretrained:
        print('use_pretrained')
        if pretrained_mode == 'imagenet':
            if in_chan == 3:
                model2d = convnext_tiny(pretrained=pretrained, ten_net=ten_net, in_chan=in_chan, use_transformer=use_transformer, **kwargs)
            else:
                model2d = convnext_tiny(pretrained=False, ten_net=ten_net, in_chan=in_chan, use_transformer=use_transformer, **kwargs)
                model2d = load_params2d(model2d, pretrained_path, ten_net, use_transformer, size='tiny')
            model3d = load_params3Dfromparams2D(model2d, model3d, init_mode, init_kind, added_dim)
        else:
            load_params3Dfromparams3D(model3d, pretrained_path, ten_net, use_transformer, size='tiny', datasize=datasize, pretrained_mode=pretrained_mode)
    return model3d


class ConvNeXt3dSTOIC(nn.Module):
    def __init__(self, modelconfig, config):
        super(ConvNeXt3dSTOIC, self).__init__()
        size = modelconfig.size if hasattr(modelconfig, 'size') else 'tiny'

        self.main_model = convnext_tiny_3d(
            pretrained=modelconfig.pretrained if hasattr(modelconfig, "pretrained") else True,
            added_dim=modelconfig.added_dim if hasattr(modelconfig, "added_dim") else 2,
            init_mode=modelconfig.init_mode if hasattr(modelconfig, 'init_mode') else 'full',
            ten_net=modelconfig.ten_net if hasattr(modelconfig, 'ten_net') else 0,
            in_chan=modelconfig.in_chan if hasattr(modelconfig, 'in_chan') else 3,
            use_transformer=modelconfig.use_transformer if hasattr(modelconfig, 'use_transformer') else False,
            pretrained_path=config.PRETRAINED_PATH if hasattr(config, 'PRETRAINED_PATH') else 'stupidpath',
            pretrained_mode=modelconfig.pretrained_mode if hasattr(modelconfig, 'pretrained_mode') else 'imagenet',
            drop_path=modelconfig.drop_path if hasattr(modelconfig, 'droppath') else 0.1,
            datasize=config.datasize if hasattr(config, 'datasize') else 256 # 256 or 128. Check, if datasize does exist in config
        )
        
        self.in_chan = modelconfig.in_chan if hasattr(modelconfig, 'in_chan') else 3
        # last_dim = self.main_model.head.in_features
        # self.main_model.head = nn.Linear(last_dim, 2)
        num_classes = getattr(modelconfig, "num_classes", 2)
        if num_classes != self.main_model.head.out_features:
            self.main_model.head = nn.Linear(
                in_features=self.main_model.head.in_features,
                out_features=num_classes,
            )

    def forward(self, x, get_features=False, train_stages=[0, 1, 2, 3]):
        if self.in_chan == 3:
            x = x.expand(-1, 3, -1, -1, -1)

        return self.main_model(x, get_features=get_features, train_stages=train_stages)



@register_model
def convnext_pretraining(**kwargs):
    model = convnext_tiny(ten_net=0, in_chan=1, use_transformer=False, **kwargs)
    return model
