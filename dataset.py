import csv
import time 
from IPython import embed
import datetime
import os
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader 
from torch.utils.data.sampler import Sampler, BatchSampler
import numpy as np
from collections import defaultdict
import itertools
import math 
# c : closing price
# h: highest price
# l: lowest price
# o: open price
# v:  volume  

# data: pre 3 days, history: 20 days now 
def get_loader_daily(dataset, batch_size, train_mode):
    instances = []
    for d, history_set in tqdm(dataset):
        history_list = list(history_set)
        history_list.sort() # sort according to date （day)

        num_days = len(history_list)
        for i in range(20, num_days): 
            if train_mode:
                if (history_list[i].year < 2017):
                    continue
            else:
                if (history_list[i].year == 2018) and (history_list[i].month > 6):
                    continue
            
            # tensor
            tensor = torch.FloatTensor(20, 5) # last 20 days  c, h, l, o, v  
            tensor[:, 0] = -1 # c 
            tensor[:, 1] = -1e10 # h 
            tensor[:, 2] = 1e10 # l 
            tensor[:, 3] = -1 # o 
            tensor[:, 4] = 0 # v

            for t in range(20): # last 20 days  c h l o v 
                index = t 
                for hour in range(9, 15):
                    for minute in range(0, 65, 5):
                        z = history_list[i-20+t] + datetime.timedelta(hours=hour, minutes=minute)   
                        if z in d:
                            tensor[index, 2] = min(tensor[index, 2].item(), d[z][2]) # l 
                            tensor[index, 1] = max(tensor[index, 1].item(), d[z][1]) # h 
                            tensor[index, 4] +=  d[z][4] # v 

                open_time = history_list[i-20+t] + datetime.timedelta(hours=9) # 9:00 as opening time 
                close_time = history_list[i-20+t] + datetime.timedelta(hours=15) # 15:00 as close time 
               
                if open_time in d :
                    tensor[index, 3] = d[open_time][3] # c

                if close_time in d:
                    tensor[index, 0] = d[close_time][0]  # o 

            # v
            v = torch.FloatTensor([0])
            # for i-th day 
            for hour in range(9, 15):
                for minute in range(0, 60, 5):
                    time_stamp = history_list[i] + datetime.timedelta(hours=hour, minutes=minute)
                    if time_stamp in d:
                        v += d[time_stamp][4]  # v
                        ticker_id = d[time_stamp][5] # ticker id 

            closing_time = history_list[i] + datetime.timedelta(hours=15) # 15:00 as close time 
            if closing_time in d:
                v += d[closing_time][4] # v 
            else:
                continue # discard data 

            if 1e9 > tensor.min().item() > -0.5:  # data validation 
                instances.append((ticker_id, tensor, v))


    if train_mode:
        random.shuffle(instances)
        l = len(instances)
        trainloader = DataLoader(instances[: int(l * 0.75)], batch_size=batch_size)
        devloader = DataLoader(instances[int(l * 0.75):], batch_size=batch_size)
        return trainloader, devloader
    else: # Test loader 
        random.shuffle(instances)
        l = len(instances)
        testloader = DataLoader(instances, batch_size=batch_size)
        return testloader


def get_loader_hourly(dataset, batch_size, train_mode):
    instances = []
    for d, history_set in tqdm(dataset):
        history_list = list(history_set)
        history_list.sort()
        num_days = len(history_list)
        for i in range(20, num_days): 
            if train_mode:
                if (history_list[i].year < 2017):
                    continue
            else:
                if (history_list[i].year == 2018) and (history_list[i].month > 6):
                    continue
            
            # tensor
            tensor = torch.FloatTensor(12, 5)
            tensor[:, 0] = -1 # c 
            tensor[:, 1] = -1e10 # h 
            tensor[:, 2] = 1e10 # l 
            tensor[:, 3] = -1 # o 
            tensor[:, 4] = 0 # v
            # last two days hourly data
            for t in range(2):
                for hour in range(9, 15):
                    index = t * 6 + hour - 9
                    y = history_list[i-2+t] + datetime.timedelta(hours=hour)
                    if y.hour == 12: # 
                        if y + datetime.timedelta(minutes=30) in d: # 12.30 下午开盘
                            tensor[index, 3] = d[y + datetime.timedelta(minutes=30)][3] # o 
                    else:
                        if y in d:
                            tensor[index, 3] = d[y][3] # o   开盘价 

                    if y.hour == 11:
                        if y + datetime.timedelta(minutes=30) in d: # 11:30 上午收盘
                            tensor[index, 0] = d[y + datetime.timedelta(minutes=30)][0] # c 
                    else:
                        if y + datetime.timedelta(minutes=55) in d:
                            tensor[index, 0] = d[y + datetime.timedelta(minutes=55)][0] # c  每个小时的收盘价 

                    for d_min in range(0, 60, 5):
                        z = y + datetime.timedelta(minutes=d_min)
                        if z in d:
                            tensor[index, 2] = min(tensor[index, 2].item(), d[z][2]) # l 
                            tensor[index, 1] = max(tensor[index, 1].item(), d[z][1]) # h 
                            tensor[index, 4] = tensor[index, 4] + d[z][4] # v 

            # history: last 20 days   c, h, l, o, v  of 9:00 - 9:55 
            history = torch.FloatTensor(20, 5)
            history[ :, 0] = -1 # c 
            history[ :, 1] = -1e10 # h 
            history[ :, 2] = 1e10 # l 
            history[ :, 3] = -1 # o 
            history[ :, 4] = 0 # v
            x = history_list[i] + datetime.timedelta(hours=9) # 9:00 
            for t in range(20):
                y = history_list[i-20+t] + datetime.timedelta(hours=9) # last 50 days 
                z = y + datetime.timedelta(minutes=55) # 9:55 
                if y in d:
                    history[t, 3] = d[y][3] # o 
                if z in d:
                    history[t, 0] = d[z][0] # c 
                for d_min in range(0, 60, 5):  # 9:00 - 9:55 : v
                    z = y + datetime.timedelta(minutes=d_min)
                    if z in d:
                        history[t, 2] = min(history[t, 2].item(), d[z][2]) # l 
                        history[t, 1] = max(history[t, 1].item(), d[z][1]) # h 
                        history[t, 4] = history[t, 4] + d[z][4] # v 

            # v
            v = torch.FloatTensor([0])
            x = history_list[i] + datetime.timedelta(hours=9)
            for minute in range(0, 65, 5):
                y = x + datetime.timedelta(minutes=minute)
                if y in d:
                    v += d[y][4]  # v
                    ticker_id = d[y][5] # ticker id 
            if (1e9 > history.min().item() > -0.5) and (1e9 > tensor.min().item() > -0.5): 
                instances.append((ticker_id, tensor, history, v))


    if train_mode:
        random.shuffle(instances)
        l = len(instances)
        trainloader = DataLoader(instances[: int(l * 0.75)], batch_size=batch_size)
        devloader = DataLoader(instances[int(l * 0.75):], batch_size=batch_size)
        return trainloader, devloader
    else: # Test loader 
        random.shuffle(instances)
        l = len(instances)
        testloader = DataLoader(instances, batch_size=batch_size)
        return testloader


def read_csv():
    if not os.path.exists('raw_data.pt'):
        train_dataset, test_dataset = [], []
        for path in tqdm(os.listdir('topix500/')):
            train_data, train_history, test_data, test_history = {}, set(), {}, set()
            
            with open('topix500/'+path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[0] == '':
                        continue

                    time_stamp = datetime.datetime.strptime(row[2] + ' ' + row[6], "%Y-%m-%d %H:%M")
                    chlov = [float(row[1]), float(row[3]), float(row[4]), float(row[5]), float(row[7])]
                    
                    if time_stamp.year == 2018:
                        test_data[time_stamp] = chlov
                        test_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))
                    else:
                        train_data[time_stamp] = chlov
                        train_history.add(datetime.datetime.strptime(row[2], "%Y-%m-%d"))
                 
            train_dataset.append((train_data, train_history))
            test_dataset.append((test_data, test_history))
            
        torch.save((train_dataset, test_dataset), 'raw_data.pt')
    else:
        train_dataset, test_dataset = torch.load('raw_data.pt')
    return train_dataset, test_dataset




def prepare_dataset(batch_size, data_name):
    if not os.path.exists("%s-cor.pt" % (data_name)):
        print('loading raw data')
        t0 = time.time()
        train_dataset, test_dataset = read_csv()
        print("loading raw data cost: ", time.time() - t0, "s") 
        print('raw data loaded. processing raw data...')
       
        if data_name == 'hourly':
            trainloader, devloader = get_loader_hourly(train_dataset, batch_size, train_mode=True)
            testloader = get_loader_hourly(test_dataset, batch_size, train_mode=False)
        elif data_name == 'daily':
            trainloader, devloader = get_loader_daily(train_dataset, batch_size, train_mode=True)
            testloader = get_loader_daily(test_dataset, batch_size, train_mode=False)
        else:
            assert(False)
        torch.save((trainloader, devloader, testloader), '%s-cor.pt' % data_name)
    else:
        print('loading from saved data pt file')
        trainloader, devloader, testloader = torch.load("%s-cor.pt" % (data_name))
    return trainloader, devloader, testloader

def test_EMA_daily(device, testloader):
    MSE, MAE, correct, cnt = 0, 0, 0, 0
    k = 0.04
    with torch.no_grad():
        for _, chlov, v in testloader:
            history = chlov 
            chlov, history, v = chlov.to(device), history.to(device), v.to(device)
            chlov, history, v = torch.log(chlov+1), torch.log(history+1), torch.log(v+1)
            output = history[:, 0, -1].exp().clone()
            for i in range(1, 20):
                output = history[:, i, -1].exp() * k + output * (1-k)
            output = output.log().view(-1, 1)
            MSE += ((output - v) ** 2).mean().item()
            MAE += ((output - v).abs()).mean().item()
            correct += ((output - chlov[:, -1, -1:]) * (v - chlov[:, -1, -1:])).ge(0).float().mean().item()
            cnt += 1
    MSE /= cnt
    MAE /= cnt
    correct /= cnt
    RMSE = math.sqrt(MSE)
    print('Test EMA: MSE: {:.6f}, RMSE: {:.6f}, MAE: {:.6f}, ACC: {:.6f} '.format(MSE, RMSE, MAE, correct))
    




if __name__ == "__main__":
    train, dev, test = prepare_dataset(32, "daily")
    print(len(train.dataset))
    print(len(dev.dataset))
    print(len(train))
    print(len(test.dataset))
    print(len(train))
    