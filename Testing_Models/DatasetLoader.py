import torch
import numpy as np

class Dataset2Class(torch.utils.data.Dataset):
    def __init__(self, path_dir:str):
        super().__init__()
        self.path_dir = path_dir
        self.data = np.genfromtxt(path_dir)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        t_data = torch.from_numpy(self.data[index][1:]).to(torch.float32)
        t_data = torch.reshape(t_data, (1, 144))
        t_label = torch.tensor(int(self.data[index][0]) - 1)
        return {'data': t_data, 'label': t_label}