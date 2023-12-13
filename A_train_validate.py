
import torch
import numpy as np
from torch.autograd import Variable
import time
from A_models import * 
import torch.nn as nn
import torch.optim as optim
import os
import time

from tqdm import tqdm
from datetime import datetime


def TrainNewGCNTransformer(train_dataloader, valid_dataloader, num_layers, num_epochs, num_heads, edge_index, edge_weight, lr, hidden_dim):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)


    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    
    New_GCNtrans = New_GCN_Transformer(num_sensors=fea_size, num_features=step_size, num_heads=num_heads, num_layers=num_layers,
                                       edge_index=edge_index, edge_weight=edge_weight,  hidden_dim = hidden_dim)
    
    New_GCNtrans = New_GCNtrans.float()  
    New_GCNtrans.to(device)

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = lr
    optimizer = torch.optim.Adam(New_GCNtrans.parameters(), lr=learning_rate)

    patience = 400
    best_valid_loss = float('inf')
    no_improvement_count = 0

    
    interval = 200
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    s_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        
        epoch_loss_train = 0
        epoch_loss_valid = 0
        num_batches_train = 0
        num_batches_valid = 0

        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue
            
            inputs, labels = inputs.to(device), labels.to(device)
           
            New_GCNtrans.zero_grad()

            Hidden_State = New_GCNtrans(inputs)
            
            loss_train = loss_MSE(Hidden_State, labels)

            
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
            epoch_loss_train += loss_train.item()
            losses_train.append(loss_train.item()) 

            num_batches_train += 1
            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                sum_loss_train = sum(losses_train[-interval:])
                loss_interval_train = np.around(sum_loss_train / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                print('Iteration #: {}, train_loss: {}, time: {}'.format(trained_number * batch_size, loss_interval_train, np.around([cur_time - pre_time], decimals=4)))

                pre_time = cur_time

        New_GCNtrans.eval()
        with torch.no_grad():
            for inputs_val, labels_val in valid_dataloader:
           
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
           
                Hidden_State = New_GCNtrans(inputs_val)
                loss_valid = loss_MSE(Hidden_State, labels_val)
                losses_valid.append(loss_valid.item())

                epoch_loss_valid += loss_valid.item()
                num_batches_valid += 1

        avg_loss_train = epoch_loss_train / num_batches_train
        avg_loss_valid = epoch_loss_valid / num_batches_valid

        if avg_loss_valid < best_valid_loss:
            best_valid_loss = avg_loss_valid
            torch.save(New_GCNtrans.state_dict(), '/home/yli3466/cs534/result/best_e7.pth')
            save_epoch = epoch+1
            saved_accuracy = best_valid_loss
        
        print(f'End of Epoch {epoch}: Average Training Loss: {avg_loss_train:.4f}, Average Validation Loss: {avg_loss_valid:.4f}')

        result_dir = "/home/yli3466/cs534/result/"
        result_file = os.path.join(result_dir, "res_gt_e7.txt")
 

        with open(result_file, "a") as f:  
            f.write(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {avg_loss_train:.4f}, Valid Loss: {avg_loss_valid:.4f}\n")
        losses_train = []
        losses_interval_train = []
        losses_valid = []
        losses_interval_valid = []

    end_time = time.time()
    total_time = end_time - s_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60       
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(result_file, "a") as f:
        f.write("="*40 + "\n")  
        f.write(f"Date and Time: {current_datetime}\n")
        f.write(f"Number of Layers: {num_layers}\n")
        f.write(f"Number of Heads: {num_heads}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Hidden Dimension: {hidden_dim}\n")
        f.write(f"Batch size: {train_dataloader.batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Best epoch: {save_epoch}\n")
        f.write(f"Best accuracy: {saved_accuracy:.4f}\n")
        f.write(f"Training finished in {total_time:.2f} seconds ({hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds).\n")
        f.write("="*40 + "\n\n")  

    return New_GCNtrans, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:

    eps = 5e-5 
    mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    
    return torch.mean(loss)

def TrainTransformer(train_dataloader, valid_dataloader, num_layers, num_epochs = 3, num_heads=3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    
    trans = Transformer(num_sensors=fea_size, num_features=step_size, num_heads=num_heads, num_layers=num_layers)
    device = "cpu"
    print("Using device:", device)
    trans = trans.to(device)

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(trans.parameters(), lr = learning_rate)
    patience = 400
    best_valid_loss = float('inf')
    no_improvement_count = 0
    use_gpu = 0
    
    interval = 10
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    s_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        epoch_loss_train = 0
        epoch_loss_valid = 0
        num_batches_train = 0
        num_batches_valid = 0

        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            
            trans.zero_grad()

            Hidden_State = trans(inputs)
            
            loss_train = masked_mse(Hidden_State, labels)

            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()

            epoch_loss_train += loss_train.item()
            num_batches_train += 1
            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                sum_loss_train = sum(losses_train[-interval:])
                loss_interval_train = np.around(sum_loss_train / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                print('Iteration #: {}, train_loss: {}, time: {}'.format(trained_number * batch_size, loss_interval_train, np.around([cur_time - pre_time], decimals=8)))

                pre_time = cur_time

        trans.eval()
        with torch.no_grad():
            for inputs_val, labels_val in valid_dataloader:
                if use_gpu:
                    inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
                else: 
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            

                Hidden_State = trans(inputs_val)
                loss_valid = loss_MSE(Hidden_State, labels_val)
                losses_valid.append(loss_valid.item())

                epoch_loss_valid += loss_valid.item()
                num_batches_valid += 1

        avg_loss_train = epoch_loss_train / num_batches_train
        avg_loss_valid = epoch_loss_valid / num_batches_valid

        if avg_loss_valid < best_valid_loss:
            best_valid_loss = avg_loss_valid
            torch.save(trans.state_dict(), '/home/yli3466/my_project_1/ML/save_model/best_Trans_ep3_model.pth')
            save_epoch = epoch+1
        print(f'End of Epoch {epoch}: Average Training Loss: {avg_loss_train}, Average Validation Loss: {avg_loss_valid}')
        result_dir = "/home/yli3466/my_project_1/ML/save_model/"
        result_file = os.path.join(result_dir, "res_Trans_ep3.txt")

        with open(result_file, "a") as f: 
            f.write(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {avg_loss_train:.4f}, Valid Loss: {avg_loss_valid:.4f}\n")
        losses_train = []
        losses_interval_train = []
        losses_valid = []
        losses_interval_valid = []

    end_time = time.time()
    total_time = end_time - s_time

    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60       
    with open(result_file, "a") as f:
        f.write("="*40 + "\n")  
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Number of patience: {patience}\n")
        f.write(f"Number of saved_model_epochs: {save_epoch}\n")
        f.write(f"Batch size: {train_dataloader.batch_size}\n")
        f.write(f"Sample size: {len(train_dataloader.dataset)}\n")  
        f.write(f"Training finished in {total_time:.2f} seconds ({hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds).\n")
    
        f.write("="*40 + "\n\n") 

    return trans, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

def TrainLSTM(train_dataloader, valid_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    lstm = LSTM(input_dim, hidden_dim, output_dim)
    
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(lstm.parameters(), lr = learning_rate)
    patience = 400
    best_valid_loss = float('inf')
    no_improvement_count = 0

    use_gpu = 0
    
    interval = 10
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []

    cur_time = time.time()
    pre_time = time.time()
    s_time = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        epoch_loss_train = 0
        epoch_loss_valid = 0
        num_batches_train = 0
        num_batches_valid = 0

        trained_number = 0
        
        lstm.train()
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            lstm.zero_grad()

            Hidden_State, Cell_State = lstm.loop(inputs)

            loss_train = loss_MSE(Hidden_State, labels)
        
            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()

            epoch_loss_train += loss_train.item()
            num_batches_train += 1

            trained_number += 1
            if trained_number % interval == 0:
                cur_time = time.time()
                sum_loss_train = sum(losses_train[-interval:])
                loss_interval_train = np.around(sum_loss_train / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                print('Iteration #: {}, train_loss: {}, time: {}'.format(trained_number * batch_size, loss_interval_train, np.around([cur_time - pre_time], decimals=8)))

                pre_time = cur_time

        lstm.eval()
        with torch.no_grad():
            for inputs_val, labels_val in valid_dataloader:
                if use_gpu:
                    inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
                else: 
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
                Hidden_State, Cell_State = lstm.loop(inputs_val)
                loss_valid = loss_MSE(Hidden_State, labels_val)
                losses_valid.append(loss_valid.item())

                epoch_loss_valid += loss_valid.item()
                num_batches_valid += 1

        avg_loss_train = epoch_loss_train / num_batches_train
        avg_loss_valid = epoch_loss_valid / num_batches_valid

        if avg_loss_valid < best_valid_loss:
            best_valid_loss = avg_loss_valid

            torch.save(lstm.state_dict(), '/home/yli3466/my_project_1/ML/save_model/best_lstm_model.pth')
            save_epoch = epoch+1
        print(f'End of Epoch {epoch}: Average Training Loss: {avg_loss_train}, Average Validation Loss: {avg_loss_valid}')
        result_dir = "/home/yli3466/my_project_1/ML/save_model"
        result_file = os.path.join(result_dir, "res_LSTM.txt")
        with open(result_file, "a") as f:  
            f.write(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {avg_loss_train:.4f}, Valid Loss: {avg_loss_valid:.4f}\n")
        losses_train = []
        losses_interval_train = []
        losses_valid = []
        losses_interval_valid = []
    end_time = time.time()
    total_time = end_time - s_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60       
    with open(result_file, "a") as f:
        f.write("="*40 + "\n")  
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Number of patience: {patience}\n")
        f.write(f"Number of saved_model_epochs: {save_epoch}\n")
        f.write(f"Batch size: {train_dataloader.batch_size}\n")
        f.write(f"Sample size: {len(train_dataloader.dataset)}\n")  
        f.write(f"Training finished in {total_time:.2f} seconds ({hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds).\n")
    
        f.write("="*40 + "\n\n") 

    return lstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

def TestLSTM(lstm, test_dataloader, max_speed):
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        Hidden_State, Cell_State = lstm.loop(inputs)
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(Hidden_State, labels)
        loss_l1 = loss_L1(Hidden_State, labels)
    
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
    
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    
    print('Tested: L1_mean: {}, L1_std : {}'.format(mean_l1, std_l1))
    return [losses_l1, losses_mse, mean_l1, std_l1]