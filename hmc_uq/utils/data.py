import numpy as np
import torch
from torch.utils.data import Dataset

class SparseDataset(Dataset):
    def __init__(self, X_sparse, Y_sparse, folding, fold, device):
        # Convert to dense matrix snd filter for fold
        X_dense = X_sparse[np.isin(folding, fold)].todense()
        Y_dense = Y_sparse[np.isin(folding, fold)].todense()

        # Filter for nonzero values
        nonzero = np.nonzero(Y_dense)[0]
        X_filtered = X_dense[nonzero]
        Y_filtered = Y_dense[nonzero]

        # Convert labels (-1 → 0)
        Y_filtered[Y_filtered == -1] = 0

        # Convert to PyTorch tensors
        self.X = torch.from_numpy(X_filtered).float().to(device)
        self.Y = torch.from_numpy(Y_filtered).to(device)

    def __len__(self):
        return len(self.X)
    
    def __getinputdim__(self):
        return self.X.shape[1]
    
    def __getdatasets__(self):
        return self.X, self.Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]