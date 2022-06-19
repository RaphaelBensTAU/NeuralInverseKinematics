import h5py
from torch.utils.data import Dataset, DataLoader
import torch


class IKDataset(Dataset):
    def __init__(self, file_path, with_orientation = False):
        self.data = h5py.File(file_path, 'r')
        self.with_orientation = with_orientation

    def __len__(self):
        return len(self.data.get('results'))

    def __getitem__(self, idx):
        positions = torch.Tensor(self.data.get('results')[idx])
        joint_angles = torch.Tensor(self.data.get('inputs')[idx])

        input = positions.squeeze(0)

        return input, joint_angles

class IKDatasetVal(Dataset):
    def __init__(self, file_path, with_orientation = False):
        self.data = h5py.File(file_path, 'r')
        self.with_orientation = with_orientation

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        positions = torch.Tensor(self.data.get('results')[len(self.data.get('results')) - idx - 1])
        joint_angles = torch.Tensor(self.data.get('inputs')[len(self.data.get('results')) - idx - 1])
        input = positions.squeeze(0)
        return input, joint_angles


