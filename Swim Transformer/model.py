import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import SwinConfig, SwinModel, Swinv2Config, Swinv2Model

class Swin(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device, patch_size, depths, input_history, num_layers=5, dropout=0.3) -> None:
        super(Swin, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_history = input_history
        self.hidden_dim = 144
        self.configuration = Swinv2Config(image_size = 72, 
                                        num_channels = self.input_history, 
                                        patch_size = self.patch_size, 
                                        embed_dim = self.hidden_dim//(2**(len(depths)-1)),
                                        depths = depths,
                                        window_size = 3, )
        self.model = Swinv2Model(self.configuration)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_dim, out_dim, device=device)

    
    def forward(self, tec, tec_target):
        output = self.model(tec)
        # print(output['last_hidden_state'].shape[:])
        # output = self.transformer_decoder(tec_target, output['last_hidden_state'])
        output = self.fc(output['last_hidden_state'])
        # print(output.shape[:])
        # input()
        return output
        
if __name__ == '__main__':
    pass