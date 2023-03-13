import argparse
import torch
import numpy as np
from dataloader import TECDataset
from train_model import train_model
from inference import inference
from torch.utils.data import DataLoader

def main(
    epoch = 100,
    batch_size = 32,
    patch_size = 12,
    target_hour = 24,
    input_history = 6,
    mode = 'None',
    device = 'cpu',
    path_save_model = 'save_model/',
):
    train_dataset = TECDataset('../data/train', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True)
    valid_dataset = TECDataset('../data/valid', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, drop_last=True, shuffle = False)
    test_dataset = TECDataset('../data/test', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    print(train_dataset.mode)
    print('done\n')

    in_dim, out_dim = 72*72, patch_size*patch_size
    model = train_model(train_dataloader, 
                        valid_dataloader,
                        in_dim, 
                        out_dim, 
                        batch_size=batch_size, 
                        patch_size=patch_size, 
                        input_history = input_history,
                        EPOCH = epoch, 
                        path_save_model = path_save_model, 
                        device = device)
    inference(model, in_dim, out_dim, test_dataloader, patch_size, input_history, device, mode, train_dataset.val, train_dataset.val2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--patch_size', '-p', type=int, default=12)
    parser.add_argument('--target_hour', '-t', type=int, default=24)
    parser.add_argument('--input_history', '-i', type=int, default=6)
    parser.add_argument('--mode', '-m', type=str, default='None')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(epoch = args.epoch, batch_size = args.batch_size, patch_size = args.patch_size, target_hour = args.target_hour, input_history = args.input_history, mode = args.mode, device = device, path_save_model = 'save_model/',)