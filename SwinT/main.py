import argparse
import torch
from dataloader import TECDataset
from train_model import train_model
from inference import inference
from model import Swin
from torch.utils.data import DataLoader

def main(
    epoch = 100,
    batch_size = 64,
    patch_size = 3,
    target_hour = 24,
    input_history = 1,
    mode = 'maxmin',
    device = 'cpu',
    path_save_model = 'save_model/',
    pretrained = False,
):
    depths = [2, 2, 2]
    if pretrained:
        train_dataset = TECDataset('../data/pretrained/train', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, depths_len = len(depths)-1)
        valid_dataset = TECDataset('../data/pretrained/valid', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, depths_len = len(depths)-1)
        test_dataset = TECDataset('../data/pretrained/test', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, depths_len = len(depths)-1)
    elif not pretrained:
        train_dataset = TECDataset('../data/train', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, depths_len = len(depths)-1)
        valid_dataset = TECDataset('../data/valid', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, depths_len = len(depths)-1)
        test_dataset = TECDataset('../data/test', mode = mode, patch_size = patch_size, target_hour = target_hour, input_history = input_history, depths_len = len(depths)-1)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, drop_last=True, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    
    print("Train year: " + train_dataset.year)
    print("Valid year: " + valid_dataset.year)
    print("Test year: " + test_dataset.year)
    print(train_dataset.mode)
    print('done\n')

    in_dim, out_dim = 72*72, patch_size*patch_size* (2**(len(depths)-1))* (2**(len(depths)-1))
    model = Swin(in_dim, out_dim, device, patch_size, depths, input_history, pretrained = pretrained).to(device)
    if not pretrained:
        model.load_state_dict(torch.load('save_model/pretrained_model.pth'))
    best_pth = train_model(model,
                        train_dataloader, 
                        valid_dataloader,
                        EPOCH = epoch, 
                        path_save_model = path_save_model, 
                        device = device,
                        pretrained = pretrained)    
    inference(model, test_dataloader, device, mode, train_dataset.val, train_dataset.val2, best_pth, pretrained = pretrained)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--patch_size', '-p', type=int, default=3)
    parser.add_argument('--target_hour', '-t', type=int, default=24)
    parser.add_argument('--input_history', '-i', type=int, default=1)
    parser.add_argument('--mode', '-m', type=str, default='maxmin')
    parser.add_argument('--pretrained', '-pt', type=bool, default=False)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(epoch = args.epoch, 
        batch_size = args.batch_size, 
        patch_size = args.patch_size, 
        target_hour = args.target_hour, 
        input_history = args.input_history, 
        mode = args.mode, device = device, 
        path_save_model = 'save_model/',
        pretrained = args.pretrained,)