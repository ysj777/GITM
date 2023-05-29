import torch
import numpy as np
import csv

def inference(model, test_dataloader, device, mode, val, val2, best_pth, pretrained):
    input, target = [], []
    model.load_state_dict(torch.load(best_pth))
    model.eval()
    total_error, step = 0, 0
    for step, batch in enumerate(test_dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        if pretrained:
            output = model(b_input)
            input_temp = [round(element.item(), 1) for sublist in output.logits[0][0][:-1] for element in sublist]
            target_temp = [round(element.item(), 1) for sublist in b_target[0][0][:-1] for element in sublist]
            input.append(input_temp)
            target.append(target_temp)
            loss = output.loss
        else:
            output = model(b_input)
            loss = reduction(np.array(output.clone().detach().cpu()), np.array(b_target.clone().detach().cpu()), mode, val, val2)
        total_error += loss.detach().item()
    save_csv(input, target)
    print("Root Mean Square Error:", total_error/step)

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

def save_csv(input, target):
    row_1 = []
    row_2 = []
    for lat in range(175, -180, -5):
        for lon in range(-180, 180, 5):
            row_1.append(lat/2)
            row_2.append(lon)
    
    with open('./predict.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(row_1)
        writer.writerow(row_2)
        for row, row2 in zip(input, target):
            writer.writerow(row)
            writer.writerow(row2)

if __name__ == '__main__':
    save_csv(None, None)