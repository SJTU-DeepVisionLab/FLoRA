import torch.nn as nn
from typing import List, Tuple
from .layers import Linear, Conv2D, Conv3D, Embedding, _MODULE_MAPPING
    
_FLoRA_MAPPING = {
    'linear': Linear,
    'conv2d': Conv2D,
    'conv3d': Conv3D,
    'embedding': Embedding,
    #TODO: add more
}

def define_flora_block(base_layer):
    flora_type = None
    for (type_name, layer_type) in _MODULE_MAPPING.items():
        if isinstance(base_layer, layer_type):
            flora_type = type_name
    return _FLoRA_MAPPING[flora_type]


class FLoRA(nn.Module):
    def __init__(self, model, target_keys=list(), base_name='model', cls_types=['linear'], **kwargs) -> None:
        super().__init__()
        assert cls_types is not None
        assert all(cn in _MODULE_MAPPING for cn in cls_types), f'cls_type error, please check support types in _MODULE_MAPPING'
        cls_types = tuple([_MODULE_MAPPING[cn] for cn in cls_types])

        #first, freeze all parameters of original model
        model.requires_grad_(False)
        self.build_FLoRA(model, target_keys=target_keys, base_name=base_name, cls_types=cls_types, **kwargs)
        self.model = model #TODO hack code. This may unsuitable for some models due to the import of 'base_model'
        #del model
        
    def build_FLoRA(self,
                    model: nn.Module, 
                    target_keys: List, 
                    base_name: str = 'model',
                    cls_types: Tuple=(nn.Linear, nn.Conv2d, nn.Embedding),
                    **kwargs):
        for name, module in model.named_children():
            full_name = base_name + '.' + name
            target_in_name = any([x in full_name for x in target_keys]) if len(target_keys)>0 else True
            if isinstance(module, cls_types) and target_in_name:
                FloraBlock = define_flora_block(module)
                setattr(model, name, FloraBlock(module, **kwargs))
            elif isinstance(module, (nn.ModuleList, nn.Sequential)) and target_in_name:
                for idx, sub_module in enumerate(module):
                    if isinstance(sub_module, cls_types):
                        FloraBlock = define_flora_block(sub_module)
                        module[idx] = FloraBlock(sub_module, **kwargs)
                    else:
                        self.build_FLoRA(sub_module, target_keys, full_name, cls_types, **kwargs)
            else:
                self.build_FLoRA(module, target_keys, full_name, cls_types, **kwargs)
                
    def forward_train(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)
    
    def forward_inference(self, inputs, **kwargs):
        #TODO: implement weight merge like peft
        NotImplementedError('TODO: implement weight merge like peft')
        
    def forward(self, inputs, mode='train', **kwargs):
        #mode: 'train', 'val', 'test'
        if mode == 'train' or mode == 'val' or mode == 'test':
            return self.forward_train(inputs, **kwargs)
        else:
            #TODO: implement weight merge like peft
            return self.forward_inference(inputs)