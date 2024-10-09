import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

class Dataloader():

    def __init__(self, data):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(data)
    

    def train_test_split(self, splitsize, data):
        # Splitting the data
        train_size = int(len(data) * splitsize) 
        train_data, test_data = data[0:train_size, :], data[train_size:len(data)+1, :]
        return train_data,test_data


    def train_valid_split(self, splitsize, train_x, train_y):
        # Split training data into training and validation sets
        train_size = int(splitsize * len(train_x))
        val_size = len(train_x) - train_size
        train_dataset, val_dataset = random_split(TensorDataset(train_x, train_y), [train_size, val_size])
        return train_dataset,val_dataset
    

    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    
    def batches(self, batch_size, train_data, val_data, test_x, test_y):
        # Create DataLoader for batching and shuffling the data
        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size, shuffle=False)
        return train_loader,val_loader,test_loader



    



