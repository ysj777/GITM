import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from transformers import ViTConfig, ViTModel

class ViT(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device, patch_size, input_history, num_layers=5, dropout=0.3) -> None:
        super(ViT, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_history = input_history
        self.hidden_dim = 144
        self.configuration = ViTConfig(image_size = 72, hidden_size= self.hidden_dim, num_channels = self.input_history, patch_size = self.patch_size)
        self.model = ViTModel(self.configuration)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_dim, out_dim, device=device)

    def forward(self, tec, tec_target):
        output = self.model(tec)
        # print(np.array(tec_target.clone().detach().cpu()).shape[:])
        # print(output['last_hidden_state'].shape[:])
        output = self.transformer_decoder(tec_target, output['last_hidden_state'][:, 1:])
        output = self.fc(output)
        # print(output.shape[:])
        return output
        
if __name__ == '__main__':
    pass