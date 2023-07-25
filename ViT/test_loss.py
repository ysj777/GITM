from plot_a_hour import process_data, plot_heatmap_on_earth_car
from model import ViT
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def create_input(truth):
    world = []
    for longitude in range(71):
        latitude = []
        for lat in range(72):
            latitude.append(truth[longitude*72 + lat])
        world.append(latitude)
    world.append([0 for _ in range(72)])
    world = [[world]]
    return torch.tensor(world, dtype=torch.float)

def main(file = 'predict',
        time = 2,
        epoch = 100,
        RECORDPATH = '.coverge_loss/',
        patch_size = 4,
        target_hour = 24,
        device = 'cpu',
        path_save_model = 'save_model/',
        target_path = '',
        pretrained = True,
        mask_ratio = 1,
        test_mode = False,
):
    
    in_dim, out_dim = 72, 72
    model = ViT(in_dim, out_dim, device, patch_size = patch_size, input_history = 1, pretrained = pretrained, mask_ratio = mask_ratio).to(device)
    if pretrained:
        best_pth = path_save_model + target_path + 'pretrained_model.pth'
    model.load_state_dict(torch.load(best_pth))
    model.eval()

    dataset = pd.read_csv(f'{file}.csv', header=list(range(2))).reset_index(drop=True)
    target = dataset.values[(time % len(dataset)) + 1]
    
    for i in range(epoch):
        truth_sr = create_input(target[2:]).to(device)
        output, mask_ = model(truth_sr)
        # print(output.logits[0][0])
        pred_sr = [round(element.item(), 1) for sublist in output.logits[0][0][:-1] for element in sublist]
        pred_sr.insert(0, model.patch_size)
        pred_sr.insert(1, str(mask_))
        truth_sr = [round(element.item(), 1) for sublist in truth_sr[0][0] for element in sublist]
        truth_sr.insert(0, target[0])
        truth_sr.insert(1, target[1])
        pred = process_data(pred_sr)
        truth = process_data(truth_sr)
        plot_heatmap_on_earth_car(np.array(pred), np.array(truth), RECORDPATH, i)
        target = pred_sr[:]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default='predict')
    parser.add_argument('--time', '-t', type=int, default=0)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--record', '-r', type=str, default='convergence_loss/')
    parser.add_argument('--patch_size', '-p', type=int, default=4)
    parser.add_argument('--target_hour', '-th', type=int, default=24)
    parser.add_argument('--pretrained', '-pt', type=bool, default=True)
    parser.add_argument('--mask_ratio', '-ma', type=int, default=1)
    parser.add_argument('--test_mode', '-tm', type=int, default=False)
   
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(file = args.file,
        time = args.time,
        epoch = args.epoch, 
        RECORDPATH = args.record,
        patch_size = args.patch_size, 
        target_hour = args.target_hour, 
        device = device, 
        path_save_model = f'save_model/',
        target_path = f'patch_{args.patch_size}_mask_ratio_{args.mask_ratio}/',
        pretrained = args.pretrained,
        mask_ratio = args.mask_ratio,
        test_mode = args.test_mode)