import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import SwinConfig, SwinModel

class Swin(nn.Module):
    def __init__(self, in_dim, out_dim, batch_size, device, patch_size, input_history, num_layers=5, dropout=0.3) -> None:
        super(Swin, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.input_history = input_history
        self.hidden_dim = 768
        # self.fc_input = nn.Linear(self.in_dim, self.hidden_dim, device=device)
        # self.fc_output = nn.Linear(self.hidden_dim, self.out_dim, device=device)
        self.configuration = SwinConfig(image_size = 72, num_channels = self.input_history, patch_size = self.patch_size, window_size = 3)
        self.model = SwinModel(self.configuration)
        self.decoder = nn.Linear(self.hidden_dim, out_dim, device=device)
    
    def forward(self, tec):
        output = self.model(tec)
        # print(output['last_hidden_state'].shape[:])
        output = self.decoder(output['last_hidden_state'])
        # print(output.shape[:])
        return output
        
if __name__ == '__main__':
    pass