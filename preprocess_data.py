import pandas as pd
import numpy as np
import pickle
from joblib import dump
data_file_path = "Gen_Data/raw_data/METR-LA/METR-LA.h5"
history_seq_len = 12
future_seq_len = 12

df = pd.read_hdf(data_file_path)  

data = np.expand_dims(df.values, axis=-1)


l, n, f = data.shape
num_samples = l - (history_seq_len + future_seq_len) + 1
test_num_short = 6850
valid_num_short = 3425
train_num_short = num_samples - valid_num_short - test_num_short

print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))

index_list = []
for t in range(history_seq_len, num_samples + history_seq_len):
    index = (t-history_seq_len, t, t+future_seq_len)
    index_list.append(index)

train_index = index_list[:train_num_short]
valid_index = index_list[train_num_short: train_num_short + valid_num_short]
test_index = index_list[train_num_short +
                        valid_num_short: train_num_short + valid_num_short + test_num_short]

from sklearn.preprocessing import StandardScaler
import numpy as np

train_end = valid_index[0][0]

train_data = data[:train_end, :, 0]

scaler = StandardScaler()
scaler.fit(train_data)
dump(scaler, 'scaler.joblib')
processed_data = scaler.transform(data[:, :, 0]).reshape(-1, 207, 1)

index = {}
index["train"] = train_index
index["valid"] = valid_index
index["test"] = test_index
filename = "./index_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len)
with open(filename, "wb") as f:
    pickle.dump(index, f)


data = {}
data["processed_data"] = processed_data
with open("./data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
    pickle.dump(data, f)

data_filename = "/Users/liyunxiao/Desktop/ML_code/data_in12_out12.pkl"

index_filename = "/Users/liyunxiao/Desktop/ML_code/index_in12_out12.pkl"

import pickle

with open(data_filename, 'rb') as f:
    data = pickle.load(f)['processed_data']

if not isinstance(data, np.ndarray):
    data = np.array(data)

np.save('processed_data.npy', data)

with open(index_filename, 'rb') as f:
    index = pickle.load(f)

train_index = np.array(index['train'])
valid_index = np.array(index['valid'])
test_index = np.array(index['test'])

np.save('train_index.npy', train_index)
np.save('valid_index.npy', valid_index)
np.save('test_index.npy', test_index)