# A helper class, it is used as an input of the DataLoader object.
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DatasetArray(Dataset):
    r"""This is a child class of the pytorch Dataset object."""
    def __init__(self, data, labels=None, transform=None):
        if labels is not None:
            self.data_arr = np.asarray(data).astype(np.float32)
            self.label_arr = np.asarray(labels).astype(np.int64)  
        else:
            tmp_arr = np.asarray(data)
            self.data_arr = tmp_arr[:, :-1].astype(np.float32)
            self.label_arr = tmp_arr[:, -1].astype(np.int64)  
        self.transform = transform
        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, index):
        data = self.data_arr[index]
        label = self.label_arr[index]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return (data, label)


# Preparation of the data for training, validation and testing a pytorch network. 
# Note that the test data is not in use for this lab.
def get_loader(batch_size =128, num_workers = 1, data=None):
    r"""This function is used to read the data file and split the data into two subsets, i.e, 
    train data and test data. Their corresponding DataLoader objects are returned."""
    
    X_train, y_train, X_test, y_test = data
    
    train_data = DatasetArray(data=X_train, labels=y_train)
    test_data = DatasetArray(data=X_test, labels=y_test)

    #The pytorch built-in class DataLoader can help us to shuffle the data, draw mini-batch,
    #do transformations, etc. 
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=100,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_loader, test_loader