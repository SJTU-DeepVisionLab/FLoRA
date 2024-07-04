import torchvision
from flora import FLoRA

def main():
    pretrained_ckpt = '/path/to/your/pretrained_ckpt/path'
    flora_params = dict(
        target_keys=['features'], #determine to append flora on the layer which contains target_key
        base_name='',   #determine the root name of the input model. 
                        #It's useful in the situation that you only want to 
                        #wrap a sub-model into the FLoRA, for example, the image encoder in LLaVA.
                        #So, base_name='vision_tower'
        cls_types=['conv2d'], #determine types to be converted to flora
        flora_cfg=dict(
            r=[16, 16, 2, 2],
            N=4,
            scale=4.0,
            drop_rate=0.01,
        ),
    )
    base_model = torchvision.models.convnext_base()
    #Note that you should implement a function to load the pretrained parameters for base_model before calling FLoRA()
    def load_pretrained_ckpt(base_model, pretrained_ckpt):
        #TODO: load your checkpoint
        return base_model
    base_model = load_pretrained_ckpt(base_model, pretrained_ckpt)

    flora_model = FLoRA(
        model=base_model,
        **flora_params
    )#.cuda()  #or any other methods to move the model from cpu to gpu to enable multi-gpu parallel training.

    print(flora_model.model)
    
if __name__ == '__main__':
    main()
