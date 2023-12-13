import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
# from Modules import FilterLinear
import math
import numpy as np
from torch_geometric.nn import GCNConv

class NewGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, edge_index, edge_weight):
        super(NewGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim)
        # self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_node_features)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_features = num_node_features

    def forward(self, x):
        #print(x.shape)
        x = x.view(-1, self.num_features)  
        x = self.conv1(x, self.edge_index, edge_weight=self.edge_weight)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index, edge_weight=self.edge_weight)
        x = F.relu(x)
        # x = self.conv3(x, self.edge_index, edge_weight=self.edge_weight)
        # x = F.relu(x)
        # x = self.conv4(x, self.edge_index, edge_weight=self.edge_weight)
        # x = F.relu(x)
        x = self.fc(x)  
        return x

class New_GCN_Transformer(nn.Module):
    def __init__(self, num_features, num_sensors, num_heads, num_layers, edge_index, edge_weight, hidden_dim):
        super(New_GCN_Transformer, self).__init__()
        self.gcn = NewGCN(num_features, hidden_dim, edge_index, edge_weight)
        
        self.positional_encoding = PositionalEncoding(d_model=num_sensors*num_features)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=num_sensors*num_features, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(num_sensors, num_sensors)

    def forward(self, src):
        #print(src.size())
        batch_size, num_features, num_sensors = src.size()

        src = self.gcn(src)

        src = src.view(-1, 1, num_sensors * num_features)
        #print(src.size())
        src = self.positional_encoding(src)
        #print(src.size())
        output = self.transformer_encoder(src)
        #print(output.size())

        output = output.view(batch_size, num_features, num_sensors )
        #print(output.size())

        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]

        return x

class Transformer(nn.Module):
    def __init__(self, num_sensors, num_features, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=num_sensors*num_features)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=num_sensors*num_features, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(num_sensors, num_sensors)


    def forward(self, src):

        batch_size, num_features, num_sensors = src.size()

        src = src.view(-1, 1, num_sensors * num_features)

        src = self.positional_encoding(src)

        output = self.transformer_encoder(src)

        output = output.view(batch_size, num_features, num_sensors )

        output = self.fc_out(output)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        super(LSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, input, Hidden_State, Cell_State):
   
        if input.dim() == 1:

            input = input.unsqueeze(0)

        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        outputs = torch.zeros(batch_size, time_step, self.hidden_size)
        for i in range(time_step):
            Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
            outputs[:, i, :] = Hidden_State
        return outputs, Cell_State

    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


