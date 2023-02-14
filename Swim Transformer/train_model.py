import torch
import logging, random
import numpy as np
from model import Swin

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

def train_model(dataloader, valid_dataloader, in_dim, out_dim, batch_size, EPOCH, path_save_model, device):
    same_seeds(1234)
    device = torch.device(device)
    model = Swin(in_dim, out_dim, batch_size, device).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)
    mse = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0
        model.train()
        for step, batch in enumerate(dataloader):            
            for param in model.parameters(): param.grad = None
            b_input, b_target = tuple(b.to(device) for b in batch[:2])
            # b_information = batch[3].to(device)
            output = model(b_input)
            
            loss = torch.sqrt(mse(output, b_target))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.detach().item()
        
        if train_loss < min_train_loss:
            torch.save(model.state_dict(), path_save_model + f'best_train_{epoch}_ViTMAE.pth')
            min_train_loss = train_loss
            best_model = f"best_train_{epoch}_ViTMAE.pth"
        
        train_loss /= len(dataloader)
        val_loss = evalute_model(valid_dataloader, model, device)
        scheduler.step(val_loss)
        logger.info(f"Epoch: {epoch:4d}, Training loss: {train_loss:5.3f}, Validation loss: {val_loss:5.3f}")
    
    return best_model

def evalute_model(dataloader, model, device):
    model.eval()    
    val_loss = 0
    mse = torch.nn.MSELoss()
    for step, batch in enumerate(dataloader):
        b_input, b_target = tuple(b.to(device) for b in batch[:2])
        # b_information = batch[3].to(device)
        # output = model(torch.cat((b_input, b_information), 2))
        output = model(b_input)
        loss = torch.sqrt(mse(output, b_target))
        val_loss += loss.detach().item()
    return val_loss / len(dataloader)
    

if __name__ == '__main__':
    pass