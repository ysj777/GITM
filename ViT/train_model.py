import torch
import logging, random
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_model(model, dataloader, valid_dataloader, EPOCH, path_save_model, device, pretrained, batch_size):
    same_seeds(1234)
    device = torch.device(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
    mse = torch.nn.MSELoss()
    min_val_loss = float('inf')
    eraly_stopping_step = 0
    val_loss_list, train_loss_list = [], []

    for epoch in range(EPOCH + 1):
        train_loss = 0
        model.train()
        for step, batch in enumerate(dataloader):            
            for param in model.parameters(): param.grad = None
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            if pretrained:
                output, _ = model(b_input)
                loss = output.loss
            else:
                output, _ = model(b_input)
                loss = torch.sqrt(mse(output.reconstruction, b_target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.detach().item()
        
        train_loss /= len(dataloader)
        val_loss = evalute_model(valid_dataloader, model, device, pretrained)
        scheduler.step(val_loss)

        train_loss_list.append(train_loss/batch_size)
        val_loss_list.append(val_loss/batch_size)

        if val_loss < min_val_loss:
            if pretrained:
                torch.save(model.state_dict(), path_save_model + 'pretrained_model.pth')
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), path_save_model + f'pretrained_model_{epoch}.pth')
            elif not pretrained:
                torch.save(model.state_dict(), path_save_model + 'best_train_ViTMAE.pth')
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), path_save_model + f'best_train_ViTMAE_{epoch}.pth')
            eraly_stopping_step = 0
            min_val_loss = val_loss
        

        if eraly_stopping_step > 20: # early stopping
            break
        eraly_stopping_step += 1

        logger.info(f"Epoch: {epoch:4d}, Training loss: {train_loss:5.3f}, Validation loss: {val_loss:5.3f}")
    show_loss(train_loss_list, val_loss_list, path_save_model)

def evalute_model(dataloader, model, device, pretrained):
    model.eval()    
    val_loss = 0
    mse = torch.nn.MSELoss()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        if pretrained:
            output, _ = model(b_input)
            loss = output.loss
        else:
            output, _ = model(b_input)
            loss = torch.sqrt(mse(output, b_target))
        val_loss += loss.detach().item()
    return val_loss / len(dataloader)

def show_loss(train_loss_list, val_loss_list, path):
    y1 = train_loss_list
    y2 = val_loss_list
    x = range(0, len(y1))
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss(MAE)') 
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend(['Train loss', 'Valid loss'])
    plt.savefig(path+'loss.png')

if __name__ == '__main__':
    pass