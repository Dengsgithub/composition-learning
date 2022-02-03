import os
import cv2
import numpy as np
import h5py
from torch.utils.data import Dataset
import random


class TrainDataset(Dataset):
    def __init__(self, dir, patch_size=64):
        super().__init__()
        self.data_path = dir
        self.patch_size = patch_size
        f = h5py.File(self.data_path, 'r')
        self.keys = list(f["label"].keys())
        random.shuffle(self.keys)
        f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        f = h5py.File(self.data_path, 'r')
        noise = f["noise"]
        input = f["input"]
        label = f["label"]
        key = self.keys[idx]
        image_b = input[key][()]
        image_o = label[key][()]
        image_n = noise[key][()]
        patch_size = self.patch_size
        h, w, c = image_b.shape
        N = np.zeros((patch_size, patch_size, c), dtype=np.float)
        if h >= patch_size and w >= patch_size:
            i = np.random.randint(h - patch_size + 1)
            j = np.random.randint(w - patch_size + 1)
            B = image_o[i:i + patch_size, j:j + patch_size]
            bilater = image_b[i:i + patch_size, j:j + patch_size]
            n = image_n[i:i + patch_size, j:j + patch_size]
            for i in range(c//3):
                N[:, :, i*3:i*3+3] = n
            RES = N - bilater
            B = np.transpose(B, (2, 0, 1))
            bilater = np.transpose(bilater, (2, 0, 1))
            RES = np.transpose(RES, (2, 0, 1))
            n = np.transpose(n, (2, 0, 1))
            sample = {'bilater': bilater, 'RES': RES, 'GT': B, 'NOISE': n}
            f.close()
            return sample


class TestDataset(Dataset):
    def __init__(self, dir):
        super().__init__()
        super().__init__()
        self.data_path = dir
        f = h5py.File(self.data_path, 'r')
        self.keys = list(f["label"].keys())
        random.shuffle(self.keys)
        f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        f = h5py.File(self.data_path, 'r')
        noise = f["noise"]
        input = f["input"]
        label = f["label"]
        key = self.keys[idx]
        image_b = input[key][()]
        image_o = label[key][()]
        image_n = noise[key][()]
        h, w, c = image_b.shape
        N = np.zeros((h, w, c), dtype=np.float)
        B = image_o
        bilater = image_b
        n = image_n

        for i in range(c // 3):
            N[:, :, i * 3:i * 3 + 3] = n
        RES = N - bilater
        B = np.transpose(B, (2, 0, 1))
        bilater = np.transpose(bilater, (2, 0, 1))
        RES = np.transpose(RES, (2, 0, 1))
        n = np.transpose(n, (2, 0, 1))
        sample = {'bilater': bilater, 'RES': RES, 'GT': B, 'NOISE': n}
        f.close()

        return sample


if __name__ == '__main__':
    dt = TrainDataset('./h5_file/0.1_0.2_0.3_0.4_0.5_10_15_3_5_.h5')
    a = dt[2]

