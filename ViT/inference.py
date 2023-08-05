import torch
import numpy as np
import csv
from tqdm import tqdm

def inference(model, test_dataloader, device, mode, val, val2, best_pth, pretrained, path):
    input, target, record = [], [], []
    model.load_state_dict(torch.load(best_pth))
    model.eval()
    mse = torch.nn.MSELoss()

    total_mae_error, total_rmse_error, step = 0, 0, 0
    for step, batch in tqdm(enumerate(test_dataloader)):
        b_info, b_input, b_target = tuple(b.to(device) for b in batch)
        if pretrained:
            output, mask_ = model(b_input)
            input_temp = [round(element.item(), 1) for sublist in output.logits[0][0][:-1] for element in sublist]
            target_temp = [round(element.item(), 1) for sublist in b_input[0][0][:-1] for element in sublist]
            
            input_temp = insert_info(input_temp, b_info, model.patch_size, mask_)
            target_temp = insert_info(target_temp, b_info, model.patch_size, mask_)
            input.append(input_temp)
            target.append(target_temp)
            mae_loss = output.loss
            rmse_error = torch.sqrt(mse(output.logits, b_target))
            record.append(mae_loss.detach().item())
        else:
            output, _ = model(b_input)
            mae_loss = 0
            rmse_error = reduction(np.array(output.clone().detach().cpu()), np.array(b_target.clone().detach().cpu()), mode, val, val2)
        total_mae_error += mae_loss.detach().item()
        total_rmse_error += rmse_error.detach().item()
    
    save_csv(input, target, path)
    print("Mean Absolute Error:", total_mae_error/step)
    print("Root Mean Square Error:", total_rmse_error/step)
    print("Standard deviation:", np.std(record))

def insert_info(arr, b_info, patch_size, mask_):
    arr.insert(0, mask_)
    arr.insert(0, patch_size)
    for info in reversed(b_info[0]):
        arr.insert(0, info.detach().item())
    return arr
    
def reduction(pred, tar, mode, val, val2):
    if mode == 'maxmin':
        pred = pred*(val-val2)+val2
        tar = tar*(val-val2)+val2
    elif mode == 'z_score':
        for i in range(len(pred)):
            pred[i] = round(pred[i]*val+val2 ,2)
            tar[i] = round(pred[i]*val+val2 ,2)
    rmse = cal_rmse(pred, tar)
    return rmse

def cal_rmse(pre, tar):
    diff = np.subtract(tar,pre)
    square = np.square(diff)
    mse = square.mean()
    rmse = np.sqrt(mse)
    return rmse

def save_csv(input, target, path):
    row_1 = ['year', 'DOY', 'hour', 'patch_size', 'mask']
    row_2 = ['year', 'DOY', 'hour', 'patch_size', 'mask']
    for lat in range(175, -180, -5):
        for lon in range(-180, 180, 5):
            row_1.append(lat/2)
            row_2.append(lon)
    
    path = path + 'predict.csv'
    with open(path, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row_1)
        writer.writerow(row_2)
        for row, row2 in zip(input, target):
            writer.writerow(row)
            writer.writerow(row2)

if __name__ == '__main__':
    save_csv(None, None)