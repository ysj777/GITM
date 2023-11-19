import argparse
import torch
import copy
from dataloader import TECDataset
from train_model import train_model
from inference import inference
from torch.utils.data import DataLoader
from model import ViT, ViT_Lora, ViT_encoder
import random
import os

def main(
    epoch = 500,
    batch_size = 32,
    patch_size = 12,
    target_hour = 1,
    input_history = 1,
    mode = 'None',
    device = 'cpu',
    path_save_model = 'save_model/',
    target_path = '',
    pretrained = False,
    mask_ratio = 1,
    test_mode = False,
    mask_type = 'random',
):
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)
    path_save_model = path_save_model + target_path
    if not os.path.isdir(path_save_model):
        os.mkdir(path_save_model)

    if pretrained:
        train_dataset = TECDataset('../data/pretrained/train', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history)
        dataset = []
        for data in train_dataset:
            dataset.append(data)
        random.shuffle(dataset)
        train_data, valid_data, test_data = dataset[:int(len(dataset)*0.8)], dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)], dataset[int(len(dataset)*0.9):]
    elif not pretrained:
        train_dataset = TECDataset('../data/train', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, pretrained = pretrained)
        dataset = []
        for data in train_dataset:
            dataset.append(data)
        random.shuffle(dataset)
        train_data, valid_data = dataset[:int(len(dataset)*0.8)], dataset[int(len(dataset)*0.8):]
        # valid_data = TECDataset('../data/valid', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, pretrained = pretrained)
        test_data = TECDataset('../data/test', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, pretrained = pretrained)
        
    train_dataloader = DataLoader(train_data, batch_size = batch_size, drop_last = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, drop_last=True, shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size = 1, shuffle = False)
    
    print("mask ratio: " + str(mask_ratio))
    print('done')
    print(f'Mask type : {mask_type}\n')

    in_dim, out_dim, hid_dim = 72, 72, 256
    model = ViT(in_dim, out_dim, hid_dim, device, patch_size, pretrained = pretrained, mask_ratio = mask_ratio, mask_type = mask_type).to(device)
    if pretrained:
        best_pth = path_save_model + 'pretrained_model.pth'
        path = path_save_model + 'pretrained.csv'
    elif not pretrained:
        model.load_state_dict(torch.load(path_save_model + 'pretrained_model.pth'))
        model = ViT_encoder(model, patch_size, hid_dim).to(device)
        best_pth = path_save_model + 'best_train_ViTMAE.pth'
        path = path_save_model + 'fine_tune.csv'

    if not test_mode:
        train_model(model,
                    train_dataloader, 
                    valid_dataloader,
                    EPOCH = epoch, 
                    path_save_model = path_save_model, 
                    device = device, 
                    pretrained = pretrained,
                    batch_size= batch_size)
    
    inference(model, test_dataloader, device, mode, train_dataset.val, train_dataset.val2, best_pth, pretrained = pretrained, path = path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=400)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--patch_size', '-p', type=int, default=4)
    parser.add_argument('--target_hour', '-t', type=int, default=1)
    parser.add_argument('--input_history', '-i', type=int, default=1)
    parser.add_argument('--mode', '-m', type=str, default='None')
    parser.add_argument('--pretrained', '-pt', type=bool, default=False)
    parser.add_argument('--mask_ratio', '-ma', type=int, default=1)
    parser.add_argument('--test_mode', '-tm', type=int, default=False)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    mask_type = 'block'

    main(epoch = args.epoch, 
         batch_size = args.batch_size, 
         patch_size = args.patch_size, 
         target_hour = args.target_hour, 
         input_history = args.input_history, 
         mode = args.mode, 
         device = device, 
         path_save_model = f'save_model/',
         target_path = f'patch_{args.patch_size}_mask_ratio_{args.mask_ratio}/',
         pretrained = args.pretrained,
         mask_ratio = args.mask_ratio,
         test_mode = args.test_mode, 
         mask_type = mask_type)