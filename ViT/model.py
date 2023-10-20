import torch.nn as nn
import numpy as np
import torch
from transformers import ViTConfig, ViTForMaskedImageModeling
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

class ViT(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, device, patch_size, pretrained = False, mask_ratio = 1) -> None:
        super(ViT, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.hidden_dim = hid_dim
        self.pretrained = pretrained
        self.mask_ratio = mask_ratio / 10

        self.configuration = ViTConfig(image_size = 72,
                                    hidden_size= self.hidden_dim,
                                    intermediate_size = self.hidden_dim*4,
                                    num_hidden_layers = 6,
                                    num_attention_heads = 8,
                                    num_channels = 1, 
                                    encoder_stride = self.patch_size,
                                    patch_size = self.patch_size,
                                    output_hidden_states = True)            
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
        self.model = model

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["decode_head"],
        )
        self.lora_model = get_peft_model(self.model, self.lora_config)

    def forward(self, tec):
        outputs = self.lora_model(tec)
        print(len(outputs[0]['hidden_states'][-1]))
        for sub_tensor in outputs[0]['hidden_states'][-1]:
            sub_tensor_shape = torch.tensor(sub_tensor).shape
            print(sub_tensor[0])
            print(len(sub_tensor[0]))
        input()
        return outputs[0]['reconstruction']

class ViT_encoder(nn.Module):
    def __init__(self, model, patch_size, hid_dim) -> None:
        super(ViT_encoder, self).__init__()
        self.patch_size = patch_size
        self.model = model
        self.output_dim = 71*72
        self.hid_dim = hid_dim
        self.num_layer = 6
        self.dropout = 0.5

        # self.freeze(self.model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward = 4096,\
                            dropout=self.dropout, norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layer)
        self.fc = nn.Linear(self.hid_dim, self.output_dim)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, tec):
        time_output = []
        for time in range(len(tec[0])):
            tec_input = tec[:, time:time+1]
            vit_ouptut = self.model(tec_input)
            vit_last_layer = vit_ouptut[0]['hidden_states'][-1][:, :1]
            time_output.append(vit_last_layer) 

        enconder_input = torch.cat(time_output, dim=1)
        time_output.clear()
        trans_output = self.transformer_encoder(enconder_input)
        trans_output = trans_output
        fc_out = F.relu(self.fc(trans_output))
        pred = fc_out[:,-1]
        return pred
    
    @staticmethod
    def freeze(model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False

if __name__ == '__main__':
    patch_size = 12
    in_dim, out_dim = 72, patch_size*patch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_history = 1
    model = ViT(in_dim, out_dim, 32, device, patch_size, input_history).to(device)