import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import datetime

import torch
import numpy as np
from torch.autograd import Variable
import time
# from Models import * 


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from A_models import * 
from A_train_validate import * 


class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_root: str, mode: str) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        
        index_file_path = os.path.join(index_root, f"{mode}_index.npy")

        self._check_if_file_exists(data_file_path, index_file_path)

        self.data = torch.from_numpy(np.load(data_file_path)).float()

        self.index = np.load(index_file_path)

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist."""
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"Cannot find data file {data_file_path}")
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError(f"Cannot find index file {index_file_path}")

    def __getitem__(self, index: int) -> tuple:

        idx = self.index[index]
        history_data = self.data[idx[0]:idx[1]].squeeze(-1)  
        future_data = self.data[idx[1]:idx[2]].squeeze(-1)
        return future_data, history_data

    def __len__(self):
        return len(self.index)


if __name__ =="__main__":
    data_file_path = "/local/scratch/yli3466/cs/processed_data.npy"
    index_root = "/local/scratch/yli3466/cs/"

    train_dataset = TimeSeriesForecastingDataset(data_file_path, index_root, mode="train")
    valid_dataset = TimeSeriesForecastingDataset(data_file_path, index_root, mode="valid")
    test_dataset = TimeSeriesForecastingDataset(data_file_path, index_root, mode="test")

    lr = 1e-5
    batch_sizes = [30,60]
    hidden_dims = [32, 128]
    num_heads_options = [3, 4]
    num_layers = [3,4]

    # lrs = [1e-5]
    # batch_sizes = [30]
    # hidden_dims = [128]
    # num_heads_options = [3]

    num_epochs = 40

    # num_layers = 3


    # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # A = np.load('/local/scratch/yli3466/cs/adj_matrix.npy')
    A = np.load('/local/scratch/yli3466/cs/adj_matrix_2.npy')
    # this is 0.0001

    rows, cols = np.where(A != 0)
    weights = A[rows, cols]

    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)

    for num_layer  in num_layers :
        for batch_size in batch_sizes:
            for hidden_dim in hidden_dims:
                for num_heads in num_heads_options:
                    print(f"Training with num_layers ={num_layers }, batch_size={batch_size}, hidden_dim={hidden_dim}, num_heads={num_heads}")

                    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
                    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
                    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

                    TrainNewGCNTransformer(train_loader, valid_loader, num_layer, num_epochs, num_heads, edge_index, edge_weight, lr, hidden_dim)

    # batch_size = 100
    # num_epochs = 100
    # num_layers = 3 
    # num_heads = 4
    # lr = 1e-4
    # hidden_dim = 128

    # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # # valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    # valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # A = np.load('/local/scratch/yli3466/cs/adj_matrix.npy')

    # rows, cols = np.where(A != 0)
    # weights = A[rows, cols]

    # edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    # edge_weight = torch.tensor(weights, dtype=torch.float)

    # TrainNewGCNTransformer(train_loader, valid_loader, num_layers, num_epochs, num_heads, edge_index, edge_weight, lr, hidden_dim)

