from dataset import prepare_dataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import random 
import argparse
import datetime
from models import MLP, LSTM, Transformer, ARTransformer, ARLSTM 
from loss import  kd_ma_loss, rkd_ma_loss_jf, akd_ma_loss 
import csv
import math
from copy import deepcopy
import wandb 

def test_mean_v20(device, testloader):
    MSE, MAE, correct, cnt = 0, 0, 0, 0
    with torch.no_grad():
        for _, chlov, history, v in testloader:
            chlov, history, v = chlov.to(device), history.to(device), v.to(device)
            chlov, history, v = torch.log(chlov+1), torch.log(history+1), torch.log(v+1)
            output = history[:, :, -1].exp().mean(dim=1).log().view(-1, 1)
            MSE += ((output - v) ** 2).mean().item()
            MAE += ((output - v).abs()).mean().item()
            correct += ((output - chlov[:, -1, -1:]) * (v - chlov[:, -1, -1:])).gt(0).float().mean().item()
            cnt += 1
    MSE /= cnt
    MAE /= cnt
    correct /= cnt
    RMSE = math.sqrt(MSE)
    print('Test mean_v20: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct), file=open(args.log, 'a'), flush=True)
    
def train(model, device, train_loader, optimizer, epoch, teacher=None, global_step=0):
    global args
    
    model.train()
    if teacher is not None: 
        teacher.eval()
    
    train_loss, cnt = 0, 0
    random.shuffle(train_loader)
    pbar = tqdm(train_loader)
    
    optimizer.zero_grad()
    cur_step = global_step 
    for chlov, history, v in pbar:
        chlov, history, v = chlov.to(device), history.to(device), v.to(device)
        chlov, history, v = torch.log(chlov+1), torch.log(history+1), torch.log(v+1)
        
        model.zero_grad()
        output = model(chlov, history)
        ar_loss = model.dist.loss(output, v).mean()   # output: (mu, sigma)

        loss = ar_loss 
        if teacher is not None: # has teacher for conducting KD
            with torch.no_grad():
                t_output = teacher(chlov, history)
            kd_loss = kd_ma_loss(teacher_ma=t_output, student_ma=output)
            rkd_loss = rkd_ma_loss_jf(teacher_ma=t_output, student_ma=output)
            akd_loss = akd_ma_loss(teacher_ma=t_output, student_ma=output)
            loss += args.kd_loss_w * kd_loss + args.rkd_loss_w * rkd_loss + args.akd_loss_w * akd_loss # add kd and rkd loss 

        loss.backward()

        train_loss += loss.item()         
        
        if cnt % args.gradient_accum == 0: # batch_size = 32 * gradient_accum
            optimizer.step()
            optimizer.zero_grad() 

        cnt += 1
        pbar.set_description("Loss %f" % (train_loss / cnt))
        cur_step += 1 

        if cur_step % args.eval_step == 0: # conduct eval here 
            pass 
    train_loss /= cnt
    print('Train Epoch: {} \tMSE: {:.6f}'.format(epoch, train_loss), file=open(args.log, 'a'), flush=True)
    return train_loss, cur_step 
    
def test(model, device, test_loader):
    model.eval()
    MSE, MAE, correct, cnt = 0, 0, 0, 0
    with torch.no_grad():
        for _, chlov, history, v in tqdm(test_loader):
            chlov, history, v = chlov.to(device), history.to(device), v.to(device)
            chlov, history, v = torch.log(chlov+1), torch.log(history+1), torch.log(v+1)
            output = model(chlov, history)
            if isinstance(output, tuple): # output is (mu, sigma)
                output = output[0]
            MSE += ((output - v) ** 2).mean().item()
            MAE += ((output - v).abs()).mean().item()
            correct += ((output - chlov[:, -1, -1:]) * (v - chlov[:, -1, -1:])).gt(0).float().mean().item()
            cnt += 1
    MSE /= cnt
    MAE /= cnt
    correct /= cnt
    RMSE = math.sqrt(MSE)
    print('Test set: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct), file=open(args.log, 'a'), flush=True)
    print('Test set: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct), flush=True)
    return MSE, RMSE, MAE, correct    
    
def valid(model, device, dev_loader):
    model.eval()
    MSE, cnt = 0, 0
    with torch.no_grad():
        for _, chlov, history, v in tqdm(dev_loader):
            chlov, history, v = chlov.to(device), history.to(device), v.to(device)
            chlov, history, v = torch.log(chlov+1), torch.log(history+1), torch.log(v+1)
            output = model(chlov, history)
            if isinstance(output, tuple): # output is (mu, sigma)
                output = output[0]
            MSE += ((output - v) ** 2).mean().item()
            cnt += 1
    MSE /= cnt
    print('Valid set: MSE: {:.6f}'.format(MSE), file=open(args.log, 'a'), flush=True)
    print('Valid set: MSE: {:.6f}'.format(MSE), flush=True)
    return MSE
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
        
def main():
    global args
    
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--full_chlov', default=True, type=str2bool)
    parser.add_argument('--log', default='', type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--dataset', default='five_minute', type=str)
    
    # learn
    parser.add_argument('--batch_size', default=32, type=int)    
    parser.add_argument('--gradient_accum', default=1, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=1.0, type=float)
    parser.add_argument('--patience', default=-1, type=float)
    parser.add_argument('--eval_step', default=200, type=float)
    
    # model
    parser.add_argument('--model', default='ARTransformer', type=str)
    parser.add_argument('--input_size', default=200, type=int)
    parser.add_argument('--hidden_size', default=200, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--attn_pooling', default=True, type=str2bool) # LSTM
    parser.add_argument('--feature_size', default=30, type=int) # LSTM
    parser.add_argument('--ar', default='gs', type=str) # Gaussian or NegativeBinary
    
    # KD
    parser.add_argument('--kd_mode', default='min', type=str)
    parser.add_argument('--teacher_path', default='', type=str)
    parser.add_argument('--teacher_num_layer', default=6, type=int)
    parser.add_argument('--kd_loss_w', default=0.0, type=float)
    parser.add_argument('--rkd_loss_w', default=0.0, type=float)
    parser.add_argument('--akd_loss_w', default=0.0, type=float)
    
    args = parser.parse_args()
    # wandb.init(project="FinKD")
    set_seed(args.seed)
    
    if args.log == '':
        args.log = datetime.datetime.now().strftime("log/%Y-%m-%d-%H:%M:%S.txt")
    print(args, file=open(args.log, 'a'), flush=True)
    
    trainloader, devloader, testloader = prepare_dataset(args.batch_size, args.dataset)

    trainloader.batch_sampler.batch_size = args.batch_size 
    devloader.batch_sampler.batch_size = args.batch_size 
    testloader.batch_sampler.batch_size = args.batch_size 
    device = torch.device("cuda")
    test_mean_v20(device, testloader)

    max_valid_MSE, max_MSE, max_RMSE, max_MAE, max_ACC = 1e10, 0, 0, 0, 0
    model_dict = { 'ARTransformer': ARTransformer, 'ARLSTM': ARLSTM}

    model = model_dict[args.model](args).to(device)
    if args.teacher_path != "":
        print("Loading teacher model")
        teacher = torch.load(args.teacher_path, map_location='cpu')
        teacher.to(device)
        print("Teacher model loading finished")
    else:
        teacher = None 

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    global_step = 0 
    early_stop = False
    patience_cnt = 0 
    patience = args.patience 
    for epoch in range(1, args.max_epoch + 1):
        if early_stop:
            print("Early Stoppping after patience")
            break 
        model.train()
        if teacher is not None: 
            teacher.eval()
        train_loss, cnt = 0, 0
        # random.shuffle(trainloader)
        pbar = tqdm(trainloader)
        
        optimizer.zero_grad()
        for _, chlov, history, v in pbar:
            chlov, history, v = chlov.to(device), history.to(device), v.to(device)
            chlov, history, v = torch.log(chlov+1), torch.log(history+1), torch.log(v+1)
            
            model.zero_grad()
            output = model(chlov, history)
            ar_loss = model.dist.loss(output, v).mean()   # output: (mu, sigma)

            loss = ar_loss 
            # wandb.log({"ar_loss": loss.item()})


            if teacher is not None: # has teacher for conducting KD
                with torch.no_grad():
                    t_output = teacher(chlov, history)
                kd_loss = kd_ma_loss(teacher_ma=t_output, student_ma=output)
                rkd_loss = rkd_ma_loss_jf(teacher_ma=t_output, student_ma=output)
                akd_loss = akd_ma_loss(teacher_ma=t_output, student_ma=output)
                loss += args.kd_loss_w * kd_loss + args.rkd_loss_w * rkd_loss  + args.akd_loss_w * akd_loss # add kd and rkd loss 
                # wandb.log({"kd_loss": kd_loss.item(), "rkd_loss": rkd_loss.item(), 'akd_loss': akd_loss.item()})

            loss.backward()
            train_loss += loss.item()         
            if cnt % args.gradient_accum == 0: # batch_size = 32 * gradient_accum
                optimizer.step()
                optimizer.zero_grad() 

            cnt += 1
            pbar.set_description("Loss %f" % (train_loss / cnt))
            global_step += 1 

            if global_step % args.eval_step == 0: # conduct eval here 
                # valid_MSE = valid(model, device, devloader)
                MSE, RMSE, MAE, ACC = test(model, device, testloader)

                if MSE < max_valid_MSE:
                    max_valid_MSE, max_MSE, max_RMSE, max_MAE, max_ACC = MSE, MSE, RMSE, MAE, ACC
                    model.cpu()            
                    torch.save(model, args.log.replace('.txt', '.pt'))
                    model.cuda()
                    patience_cnt = 0
                else:
                    patience_cnt += 1 
                model.train() 
                if patience > 0 and patience_cnt > patience:
                    early_stop = True  
                    break 
               
        scheduler.step()
        train_loss /= cnt
        print('Train Epoch: {} \tMSE: {:.6f}'.format(epoch, train_loss), file=open(args.log, 'a'), flush=True)
        
    f = open('deep_ar_ret_teacher.csv', 'a', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow([args.dataset, args.model, 'num_layer',args.num_layer, 'ar mode: ', 'cor: ', args.cor, args.ar, 'kdw: ', args.kd_loss_w, 'rkd_w: ', args.rkd_loss_w, 'akd_w: ', args.akd_loss_w, 'seed: ', args.seed, 'ret: ',max_MSE, max_RMSE, max_MAE, max_ACC])
    f.close()
    
main()
