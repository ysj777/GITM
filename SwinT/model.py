import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import SwinConfig, SwinModel
from transformers import Swinv2Config, Swinv2Model, Swinv2ForMaskedImageModeling

class Swin(nn.Module):
    def __init__(self, in_dim, out_dim, device, patch_size, depths, input_history, num_layers=5, pretrained = False) -> None:
        super(Swin, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.depths = depths
        self.input_history = input_history
        self.hidden_dim = 144
        self.pretrained = pretrained

        self.configuration = Swinv2Config(image_size = 72, 
                                        num_channels = self.input_history, 
                                        patch_size = self.patch_size, 
                                        embed_dim = self.hidden_dim//(2**(len(depths)-1)),
                                        depths = depths,
                                        encoder_stride = self.patch_size*4,
                                        window_size = 3, )
        self.Swin_mask = Swinv2ForMaskedImageModeling(self.configuration)
        self.Swin = Swinv2Model(self.configuration)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_dim, out_dim, device=device)

    def forward(self, tec):
        if self.pretrained:
            num_patches = (self.Swin_mask.config.image_size // self.Swin_mask.config.patch_size) ** 2
            bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool().to(self.device)
            outputs = self.Swin_mask(tec, bool_masked_pos)
            return outputs
        elif not self.pretrained:
            output = self.Swin_mask(tec)
            # print(output['last_hidden_state'].shape[:])
            # output = self.transformer_decoder(tec_target, output['last_hidden_state'])
            output = self.fc(output['last_hidden_state'])
            return output
        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depths = [2, 2, 2]
    patch_size = 3
    out_dim = patch_size*patch_size* (2**(len(depths)-1))* (2**(len(depths)-1))
    model = Swin(72*72, out_dim, device, patch_size, depths, 1, pretrained=True).to(device)