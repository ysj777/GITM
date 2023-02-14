import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import SwinConfig, SwinModel

class Swin(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device, num_layers=5, dropout=0.3) -> None:
        super(Swin, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.hidden_dim = 768
        # self.fc_input = nn.Linear(self.in_dim, self.hidden_dim, device=device)
        # self.fc_output = nn.Linear(self.hidden_dim, self.out_dim, device=device)
        self.configuration = SwinConfig(image_size = 72, num_channels = 1, patch_size = 4, window_size = 3)
        self.model = SwinModel(self.configuration)
        self.decoder = nn.Linear(self.hidden_dim, out_dim, device=device)
    
    def forward(self, tec):
        output = self.model(tec)
        print(output['last_hidden_state'].shape[:])
        print(output)
        output = self.decoder(output['last_hidden_state'])
        print(output)
        return output
        
if __name__ == '__main__':
    pass