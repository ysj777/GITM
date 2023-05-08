import torch
import numpy as np

def inference(model, test_dataloader, device, mode, val, val2, best_pth, pretrained):
    tec_tar, tec_pred = [], []
    model.load_state_dict(torch.load('save_model/' + best_pth))
    model.eval()
    total_rmse, step = 0, 0
    for step, batch in enumerate(test_dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        # b_information = batch[3].to(device)
        # b_time = tuple(b.numpy() for b in batch[2])
        if pretrained:
            output = model(b_input)
            output = output.logits
        else:
            output = model(b_input)
        rmse = reduction(np.array(output.clone().detach().cpu()), np.array(b_target.clone().detach().cpu()), mode, val, val2)
        total_rmse += rmse
    print("Root Mean Square Error:", total_rmse/step)

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

if __name__ == '__main__':
    pass