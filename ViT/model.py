import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from transformers import ViTConfig, ViTModel

class ViT(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device, patch_size, input_history, num_layers=4, dropout=0.3) -> None:
        super(ViT, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_history = input_history
        self.hidden_dim = 144
        self.embedding = nn.Linear(self.in_dim, self.hidden_dim, device=device)
        self.configuration = ViTConfig(image_size = 72,
                                    hidden_size= self.hidden_dim,
                                    intermediate_size = self.hidden_dim*4,
                                    num_channels = self.input_history, 
                                    patch_size = self.patch_size)
        self.ViT_encoder = ViTModel(self.configuration)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_dim, self.out_dim, device=device)

    def forward(self, tec, tec_target):
        memory = self.ViT_encoder(tec)
        # print(np.array(tec.clone().detach().cpu()).shape[:])
        # print(memory['last_hidden_state'].shape[:])
        tgt_embedded = self.embedding(tec)
        tgt_mask = self._generate_square_subsequent_mask(tgt_embedded.size(1)).to(self.device)
        print(np.array(tec_target.clone().detach().cpu()).shape[:])
        output = self.transformer_decoder(tec_target, memory['last_hidden_state'][:, 1:], tgt_mask = tgt_mask)
        output = self.fc(output)
        # print(output.shape[:])
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)==1).transpose(0, 1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
if __name__ == '__main__':
    pass