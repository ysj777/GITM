import torch
import numpy as np
from model import Swin

def inference(best_model, in_dim, out_dim, test_dataloader, patch_size, depths, input_history, device, mode, val, val2):
    tec_tar, tec_pred = [], []
    model = Swin(in_dim, out_dim, 1, device, patch_size, depths, input_history).to(device)
    model.load_state_dict(torch.load('save_model/'+ best_model))
    model.eval()
    total_rmse, step = 0, 0
    for step, batch in enumerate(test_dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        # b_information = batch[3].to(device)
        # b_time = tuple(b.numpy() for b in batch[2])
        output = model(b_input)
        rmse = reduction(np.array(output.clone().detach().cpu()), np.array(b_target.clone().detach().cpu()), mode, val, val2)
        total_rmse += rmse
    print("Root Mean Square Error:", total_rmse/step)

def reduction(pred, tar, mode, val, val2):
    if mode == 'maxmin':
        for i in range(len(pred)):
            pred[i] = pred[i]*(val-val2)+val2
            tar[i] = tar[i]*(val-val2)+val2
    elif mode == 'z_score':
        for i in range(len(pred)):
            pred[i] = round(pred[i]*val+val2 ,2)
            tar[i] = round(pred[i]*val+val2 ,2)
    rmse = cal_rmse(pred, tar)
    return rmse

def cal_rmse(pre, tar):
    diff=np.subtract(tar,pre)
    square=np.square(diff)
    mse=square.mean()
    rmse=np.sqrt(mse)
    return rmse

if __name__ == '__main__':
    pass