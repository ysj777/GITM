import torch.nn as nn
import numpy as np
import torch
from transformers import ViTConfig, ViTForMaskedImageModeling
from peft import LoraConfig, get_peft_model

class ViT(nn.Module):
    def __init__(self, in_dim, out_dim, device, patch_size, input_history, num_layers=4, pretrained = False, mask_ratio = 1) -> None:
        super(ViT, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.input_history = input_history
        self.hidden_dim = 72
        self.pretrained = pretrained
        self.mask_ratio = mask_ratio / 9

        self.configuration = ViTConfig(image_size = 72,
                                    # hidden_size= self.hidden_dim,
                                    # intermediate_size = self.hidden_dim*4,
                                    num_channels = self.input_history, 
                                    encoder_stride = self.patch_size,
                                    patch_size = self.patch_size,)            
        self.ViT_mask = ViTForMaskedImageModeling(self.configuration)

    def forward(self, tec):
        num_patches = (self.ViT_mask.config.image_size // self.ViT_mask.config.patch_size) ** 2
        num_masked_patches = int(num_patches * self.mask_ratio)
        masked_indices = torch.randperm(num_patches)[:num_masked_patches]
        bool_masked_pos = torch.zeros(size=(1, num_patches),dtype=torch.bool, device = self.device)
        bool_masked_pos[:, masked_indices] = True
        outputs = self.ViT_mask(tec, bool_masked_pos)
        return outputs, list(np.array(masked_indices))
    
class ViT_Lora(nn.Module):
    def __init__(self, model, patch_size) -> None:
        super(ViT_Lora, self).__init__()
        self.patch_size = patch_size
        self.moodel = model

        self.lora_config = LoraConfig(
            r=24,
            lora_alpha=12,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["decode_head"],
        )
        self.lora_model = get_peft_model(self.moodel, self.lora_config)
        self.print_trainable_parameters(self.lora_model)

    def forward(self, tec):
        outputs = self.lora_model(tec)
        return outputs[0]['reconstruction']

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    
if __name__ == '__main__':
    patch_size = 12
    in_dim, out_dim = 72, patch_size*patch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_history = 1
    model = ViT(in_dim, out_dim, 32, device, patch_size, input_history).to(device)