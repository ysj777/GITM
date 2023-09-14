import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torch
from transformers import ViTConfig, ViTModel, ViTForMaskedImageModeling
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

        # self.embedding = nn.Linear(self.in_dim, self.hidden_dim, device=device)
        self.configuration = ViTConfig(image_size = 72,
                                    # hidden_size= self.hidden_dim,
                                    # intermediate_size = self.hidden_dim*4,
                                    num_channels = self.input_history, 
                                    encoder_stride = self.patch_size,
                                    patch_size = self.patch_size,)            
        self.ViT_mask = ViTForMaskedImageModeling(self.configuration)
        # self.ViT = ViTModel(self.configuration)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead=8)
        # self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # self.fc = nn.Linear(self.hidden_dim, self.out_dim, device=device)

        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["decode_head"],
        )
        self.lora_model = get_peft_model(copy.deepcopy(self.ViT_mask), self.lora_config)


    def forward(self, tec):
        if self.pretrained:
            num_patches = (self.ViT_mask.config.image_size // self.ViT_mask.config.patch_size) ** 2
            num_masked_patches = int(num_patches * self.mask_ratio)
            masked_indices = torch.randperm(num_patches)[:num_masked_patches]
            bool_masked_pos = torch.zeros(size=(1, num_patches),dtype=torch.bool, device = self.device)
            bool_masked_pos[:, masked_indices] = True
            outputs = self.ViT_mask(tec, bool_masked_pos)
            return outputs, list(np.array(masked_indices))
        elif not self.pretrained:
            # print(np.array(tec.clone().detach().cpu()))
            outputs = self.lora_model(tec)
            return outputs['reconstruction']

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
if __name__ == '__main__':
    patch_size = 12
    in_dim, out_dim = 72, patch_size*patch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_history = 1
    model = ViT(in_dim, out_dim, 32, device, patch_size, input_history).to(device)