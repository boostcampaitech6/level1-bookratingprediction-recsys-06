import os
import tqdm
import wandb
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam, AdamW, NAdam, SparseAdam
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, MultiStepLR,  \
    ExponentialLR, CosineAnnealingLR, CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


def train(args, model, dataloader, logger, setting):
    earlystop_cnt = 0
    minimum_loss = 999999999
    if args.loss_fn == 'MSE':
        loss_fn = MSELoss()
    elif args.loss_fn == 'RMSE':
        loss_fn = RMSELoss()
    else:
        pass
    
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'NAdam':
        optimizer = NAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SparseAdam':
        optimizer = SparseAdam(model.parameters(), lr=args.lr)
    else:
        pass
    
    if args.scheduler == 'LAMBDA':
        scheduler = LambdaLR(optimizer, lrlambda=lambda epoch: 0.65 ** epoch)
    elif args.scheduler == 'Multiplicative':
        scheduler = MultiplicativeLR(optimizer, lrlambda=lambda epoch: 0.65 ** epoch)
    elif args.scheduler == 'Step':
        scheduler = StepLR(optimizer, stepsize=2, gamma=0.1)
    elif args.scheduler == 'MultiStep':
        scheduler = MultiStepLR(optimizer, milestones=[6,8,9], gamma=0.1)
    elif args.scheduler == 'Exponential':
        scheduler = ExponentialLR(optimizer, gamma=0.1)
    elif args.scheduler == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer, Tmax=10, eta_min=0)
    elif args.scheduler == 'Cyclic_triangular':
        scheduler = CyclicLR(optimizer, baselr=0.001, maxlr=0.1, stepsizeup=5, stepsize_down=5, mode="triangular")
    elif args.scheduler == 'Cyclic_triangular2':
        scheduler = CyclicLR(optimizer, baselr=0.001, maxlr=0.1, stepsizeup=5, stepsize_down=5, mode="triangular2")
    elif args.scheduler == 'OneCycle_cos':
        scheduler = OneCycleLR(optimizer, maxlr=0.1, stepsperepoch=10, epochs=10, anneal_strategy='cos')
    elif args.scheduler == 'OneCycle_linear':
        scheduler = OneCycleLR(optimizer, maxlr=0.1, stepsperepoch=10, epochs=10, anneal_strategy='linear')
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
    else:
        pass
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)
    
    for epoch in tqdm.tqdm(range(args.epochs)):
        model.train()
        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['train_dataloader']):
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch +=1
        valid_loss = valid(args, model, dataloader, loss_fn)
        logger.log(epoch=epoch+1, train_loss=total_loss/batch, valid_loss=valid_loss)
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            earlystop_cnt = 0
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
        else : 
            earlystop_cnt += 1
        
        wandb.log({'Train Loss' : total_loss/batch, 'Valid Loss' : valid_loss, 'Best Loss' : minimum_loss})
        print(f' Epoch : {epoch+1}, Train Loss : {total_loss/batch:.3f}, Valid Loss : {valid_loss:.3f}, LR : {args.lr}, EarlyStop Count : {earlystop_cnt}')
        
        if earlystop_cnt >= args.patience :
            print(f' Early Stopping. Best Valid_loss : {minimum_loss:.3f}')
            break
    wandb.finish()
    logger.close()
    return model
        

def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for idx, data in enumerate(dataloader['valid_dataloader']):
        if args.model == 'CNN_FM':
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
    valid_loss = total_loss/batch
    return valid_loss


def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt'))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts
