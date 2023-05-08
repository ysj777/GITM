import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from transformers import ViTConfig, ViTModel, ViTForMaskedImageModeling

class ViT(nn.Module):
    def __init__(self, in_dim, out_dim, device, patch_size, input_history, num_layers=4, pretrained = False) -> None:
        super(ViT, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.input_history = input_history
        self.hidden_dim = 72
        self.pretrained = pretrained

        # self.embedding = nn.Linear(self.in_dim, self.hidden_dim, device=device)
        self.configuration = ViTConfig(image_size = 72,
                                    # hidden_size= self.hidden_dim,
                                    # intermediate_size = self.hidden_dim*4,
                                    num_channels = self.input_history, 
                                    encoder_stride = self.patch_size,
                                    patch_size = self.patch_size,)            
        self.ViT_mask = ViTForMaskedImageModeling(self.configuration)
        self.ViT = ViTModel(self.configuration)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_dim, self.out_dim, device=device)

    def forward(self, tec):
        if self.pretrained:
            num_patches = (self.ViT_mask.config.image_size // self.ViT_mask.config.patch_size) ** 2
            bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool().to(self.device)
            outputs = self.ViT_mask(tec, bool_masked_pos)
            return outputs
        elif not self.pretrained:
            # print(np.array(tec.clone().detach().cpu()))
            output = self.ViT_mask(tec)
            # print(output['logits'])
            output = self.fc(output['logits'])
            return output

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