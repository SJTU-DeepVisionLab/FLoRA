import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

_MODULE_MAPPING = {
    'linear': nn.Linear,
    'conv2d': nn.Conv2d,
    'conv3d': nn.Conv3d,
    'embedding': nn.Embedding,
    #TODO: add more
}


class FLoRALayer(nn.Module):
    def __init__(self,
                 layer_type: str,
                 #========= channels ===========
                 #dims should be [in, out, other]
                 dims: List = [256, 512],
                 #kernel_size: Union[int, List] = 1,
                 #========== flora configs ============
                 N: int = 2,
                 r: List = [8, 8],
                 scale: float = 1.0,
                 #========== kwargs ===========
                 **kwargs) -> None:
        super().__init__()
        assert N > 1, f'The minimum N_dims should be 1'
        assert all(r) >= 1, f'The minimum r should be 1'
        assert len(r) == len(dims), f'The length of r and dims should be same'
        assert N == len(r), f'The dimension of List "r" should equal to N_dims'
            
        self.layer_type = layer_type
        self.dims = dims
        self.N = N
        self.r = r
        self.scale = scale
        
        # define G and each components
        self.register_parameter('param_G',nn.Parameter(torch.randn(self.r)))
        for i in range(self.N):
            self.register_parameter('param_{}'.format(i),
                                    nn.Parameter(
                                        torch.randn(self.dims[i], self.r[i])
                                    ))
            
    def init_parameters(self):
        #nn.init.kaiming_normal(self.param_G, a=math.sqrt(3))
        for i in range(self.N):
            nn.init.kaiming_normal_(
                getattr(
                    self, 
                    'param_{}'.format(i)
                ),
                a=math.sqrt(3)
            )
        nn.init.zeros_(self.param_G)
        
    def forward(self, style='v2', src_weight_mean=1.0):
        if style == 'v2':
            dt_weight = torch.tensordot(self.param_G, self.param_0, dims=[[0], [1]])
            for i in range(1, self.N):
                dt_weight = torch.tensordot(
                    dt_weight, 
                    getattr(self, 'param_{}'.format(i)),
                    dims=[[0], [1]]
                )
        if dt_weight.dim() > 1:
            dt_weight = dt_weight.transpose(0, 1)
        dt_weight = self.scale * dt_weight
        
        return dt_weight
    
    def __repr__(self):
        s = f"FLoRALayer(\n"
        s +=f"type={self.layer_type}, N={self.N}, r={self.r}, scale={self.scale}, \n"
        s +=f"param_G: {self.param_G.shape}, \n"
        for i in range(self.N):
            s += f"param_{i}: {getattr(self, 'param_{}'.format(i)).shape}, \n"
        return s
    
class BaseFLoRA(nn.Module):
    def __init__(self, 
                base_layer: nn.Module, 
                flora_cfg: Dict,
                base_name: str = 'model',
                **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.base_layer = base_layer
        self.N = flora_cfg['N']
        self.r = flora_cfg['r'] if isinstance(flora_cfg['r'], List) else [flora_cfg['r'] for _ in range(self.N)]
        self.scale = flora_cfg['scale']
        self.set_dims()
        
        self.make_flora_layer()
        self.flora_dropout = nn.Dropout(p=flora_cfg['drop_rate']) if flora_cfg['drop_rate'] > 0 else nn.Identity()
        self.layer_list = ['base_layer', 'flora_layer']
        
        self.init_parameters()
        self.disable_base_weight_grad()
        
    
    def set_dims(self,): #default for linear
        self.in_dims = [self.base_layer.in_features]
        self.out_dims = [self.base_layer.out_features]
        self.other_dims = getattr(self, 'other_dims', [])
        
    def init_parameters(self):
        for layer_name in self.layer_list:
            layer = getattr(self, layer_name, None)
            if getattr(layer, 'init_parameters', None) is not None:
                layer.init_parameters()
                
    def disable_base_weight_grad(self):
        self.base_layer.requires_grad_(False)

    def make_flora_layer(self, **kwargs):
        self.flora_layer = FLoRALayer(
            layer_type=self.layer_type,
            dims=self.in_dims + self.out_dims + self.other_dims,
            N=self.N,
            r=self.r,
            scale=self.scale,
            **kwargs
        )
        
    def get_new_weight(self):
        return self.flora_layer.forward(style='v2')
    
class Linear(BaseFLoRA):
    def __init__(self, base_layer: nn.Module, flora_cfg: Dict, **kwargs) -> None:
        self.layer_type = 'linear'
        self.other_dims = []
        assert getattr(base_layer, 'in_features', 0) > 0
        assert getattr(base_layer, 'out_features', 0) > 0
        assert flora_cfg['N'] == 2
        assert len(flora_cfg['r']) == 2
        
        super().__init__(base_layer, flora_cfg, **kwargs)
        
    def forward(self, inputs): #for linear
        base_out = self.base_layer(inputs)
        
        new_weight = self.get_new_weight().to(inputs.dtype)
        flora_out = F.linear(
            self.flora_dropout(inputs),
            weight=new_weight
        )
        
        out = base_out + flora_out
        
        return out

class Conv2D(BaseFLoRA):
    def __init__(self, base_layer: nn.Module, flora_cfg: Dict, **kwargs) -> None:
        self.layer_type = 'conv2d'
        assert getattr(base_layer, 'in_channels', 0) > 0
        assert getattr(base_layer, 'out_channels', 0) > 0
        assert flora_cfg['N'] == 4
        assert len(flora_cfg['r']) == 4
        
        super().__init__(base_layer, flora_cfg, **kwargs)
        
    def set_dims(self):
        self.in_dims = [self.base_layer.in_channels]
        self.out_dims = [self.base_layer.out_channels]
        self.other_dims = list(self.base_layer.kernel_size)
        
        self.base_stride = self.base_layer.stride
        self.base_padding = self.base_layer.padding
        self.base_dilation = self.base_layer.dilation
        self.base_groups = self.base_layer.groups
        
    def forward(self, inputs): #for linear
        base_out = self.base_layer(inputs)
        
        new_weight = self.get_new_weight().to(inputs.dtype)
        flora_out = F.conv2d(
            self.flora_dropout(inputs),
            weight=new_weight,
            stride=self.base_stride,
            padding=self.base_padding,
            dilation=self.base_dilation,
            groups=self.base_groups
        )
        
        out = base_out + flora_out
        
        return out

class Conv3D(BaseFLoRA):
    def __init__(self, base_layer: nn.Module, flora_cfg: Dict, **kwargs) -> None:
        self.layer_type = 'conv3d'
        assert getattr(base_layer, 'in_channels', 0) > 0
        assert getattr(base_layer, 'out_channels', 0) > 0
        assert flora_cfg['N'] == 5
        assert len(flora_cfg['r']) == 5
        
        super().__init__(base_layer, flora_cfg, **kwargs)
        
    def set_dims(self):
        self.in_dims = [self.base_layer.in_channels]
        self.out_dims = [self.base_layer.out_channels]
        self.other_dims = list(self.base_layer.kernel_size)
        
        self.base_stride = self.base_layer.stride
        self.base_padding = self.base_layer.padding
        self.base_dilation = self.base_layer.dilation
        self.base_groups = self.base_layer.groups
        
    def forward(self, inputs): #for linear
        base_out = self.base_layer(inputs)
        
        new_weight = self.get_new_weight().to(inputs.dtype)
        flora_out = F.conv3d(
            self.flora_dropout(inputs),
            weight=new_weight,
            stride=self.base_stride,
            padding=self.base_padding,
            dilation=self.base_dilation,
            groups=self.base_groups
        )
        
        out = base_out + flora_out
        
        return out
    
class Embedding(BaseFLoRA):
    def __init__(self, base_layer: nn.Module, flora_cfg: Dict, **kwargs) -> None:
        self.layer_type = 'embedding'
        assert getattr(base_layer, 'embedding_dim', 0) > 0
        assert flora_cfg['N'] == 1
        assert len(flora_cfg['r']) == 1
        
        super().__init__(base_layer, flora_cfg, **kwargs)

    def set_dims(self):
        self.in_dims = [1]
        self.out_dims = [self.base_layer.embedding_dim]
        self.other_dims = []

    def forward(self, inputs):
        base_out = self.base_layer(inputs)
        
        new_weight = self.get_new_weight().to(base_out.dtype)
        flora_out = new_weight.repeat(base_out.shape[0], 1)
        
        out = base_out + flora_out
        
        return out