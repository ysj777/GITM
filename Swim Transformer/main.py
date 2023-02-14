import argparse
import torch
from dataloader import TECDataset
from train_model import train_model
from inference import inference
from torch.utils.data import DataLoader

def main(
    window_size = 24,
    epoch = 100,
    batch_size = 64,
    mode = 'None',
    device = 'cpu',
    path_save_model = 'save_model/',
):
    train_dataset = TECDataset('../data/train', mode = mode, window_size = window_size)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True)
    valid_dataset = TECDataset('../data/valid', mode = mode, window_size = window_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, drop_last=True, shuffle = False)
    # test_dataset = TECDataset('../data/test', mode = mode, window_size = window_size)
    # test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    # print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    print(train_dataset.mode)
    print('done\n')

    in_dim, out_dim = 71*72, 72
    model = train_model(train_dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH = epoch, path_save_model = path_save_model, device = device)
    # inference(model, in_dim, out_dim, test_dataloader, device, mode, train_dataset.val, train_dataset.val2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--mode', '-m', type=str, default='None')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    main(epoch = args.epoch, batch_size = args.batch_size, mode = args.mode, device = device, path_save_model = 'save_model/',)