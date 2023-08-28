from plot_a_hour import plot_heatmap_on_earth_car
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

def process_data(pred, truth):
    info = pred[:3]
    patch_size = pred[3]
    temp = pred[4][1:-1]
    mask = list(map(int, temp.split(',')))
    pred_tec_data = pred[5:]
    truth_tec_data = truth[5:]
    patch_count = 72*72//(patch_size*patch_size)
    target_world = [[]for _ in range(patch_count)]
    pred_and_truth = [[]for _ in range(patch_count)]

    for longitude in range(71):
        for lat in range(72):
            patch_idx = (longitude//patch_size)*(72//patch_size) + lat//patch_size
            if patch_idx in mask:
                target_world[patch_idx].append(pred_tec_data[longitude*72 + lat])
                pred_and_truth[patch_idx].append(pred_tec_data[longitude*72 + lat])
            else:
                target_world[patch_idx].append(-1)
                pred_and_truth[patch_idx].append(truth_tec_data[longitude*72 + lat])

    tec_map = []
    pred_plus_truth = []
    for patch in range(0, len(target_world), 72//patch_size):
        for lat_idx in range(len(target_world[patch])//patch_size):
            for lon_idx in range(72//patch_size):
                tec_map += target_world[patch + lon_idx][lat_idx*patch_size:(lat_idx+1)*patch_size]
                pred_plus_truth += pred_and_truth[patch + lon_idx][lat_idx*patch_size:(lat_idx+1)*patch_size]
    return info, tec_map, pred_plus_truth

def insert_info(arr, b_info, patch_size, mask_):
    arr.insert(0, mask_)
    arr.insert(0, patch_size)
    for info in reversed(b_info):
        arr.insert(0, info)
    return arr

def main(file = 'predict',
        time = 0,
        epoch = 100,
        RECORDPATH = '.coverge_loss/',
        patch_size = 4,
        target_hour = 24,
        device = 'cpu',
        path_save_model = 'save_model/',
        target_path = '',
        pretrained = True,
        mask_ratio = 1,
):
    
    in_dim, out_dim = 72, 72
    model = ViT(in_dim, out_dim, device, patch_size = patch_size, input_history = 1, pretrained = pretrained, mask_ratio = mask_ratio).to(device)
    if pretrained:
        best_pth = path_save_model + target_path + 'pretrained_model.pth'
    model.load_state_dict(torch.load(best_pth))
    model.eval()

    dataset = pd.read_csv(f'{file}.csv', header=list(range(2))).reset_index(drop=True)
    target = dataset.values[(2*time % len(dataset)) + 1]
    temp_origin_sr = create_input(target[5:]).to(device)
    temp_origin_sr = [round(element.item(), 1) for sublist in temp_origin_sr[0][0] for element in sublist]

    for i in range(epoch):
        truth_sr = create_input(target[5:]).to(device)
        output, mask_ = model(truth_sr)
        pred_sr = [round(element.item(), 1) for sublist in output.logits[0][0][:-1] for element in sublist]
        truth_sr = [round(element.item(), 1) for sublist in truth_sr[0][0] for element in sublist]
        pred_sr = insert_info(pred_sr, target[:3], model.patch_size, str(mask_))
        truth_sr = insert_info(truth_sr, target[:3], model.patch_size, str(mask_))
        
        origin_sr = temp_origin_sr[:]
        origin_sr = insert_info(origin_sr, target[:3], model.patch_size, str(mask_))
        
        p_info, pred_sr, next_truth = process_data(pred_sr, truth_sr)
        t_info, origin_sr, _ = process_data(origin_sr, origin_sr)
        plot_heatmap_on_earth_car(np.array(origin_sr), np.array(pred_sr), RECORDPATH, i, p_info)
        next_truth = insert_info(next_truth, target[:3], model.patch_size, str(mask_))
        target = next_truth[:]
        

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
        mask_ratio = args.mask_ratio)