# Name: Jiaming Yu
# Time:
from genericpath import exists
from operator import index
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms
import numpy as np
from model import MyNBVNetV3
import pickle


class VOXELDataset(Dataset):
    def __init__(self, path, transform=None, processed_data=True, save_path='./train_36.dat'):
        self.path = path
        self.processed_data = processed_data
        self.save_path = save_path
        self.transform = transform

        # print(len(self.grid_path))
        if self.processed_data:
            self.grid_path = []
            self.label_path = []

            for root, _, files in os.walk(self.path):
                for file in files:
                    if "grid" in file:
                        grid_path = os.path.join(root, file)
                        self.grid_path.append(grid_path)
                    else:
                        label_path = os.path.join(root, file)
                        self.label_path.append(label_path)

            if not os.path.exists(self.save_path):
                print('processing data, only running in the first epoch')
                self.list_of_grid = [None] * len(self.grid_path)
                self.list_of_label = [None] * len(self.label_path)
                #print('save_path:', self.save_path, '\n')
                for index in range(len(self.grid_path)):
                    grid_path = self.grid_path[index]
                    label_path = self.label_path[index]
                    #print('grid_path:', grid_path, "\n", 'label_path:', label_path)
                    grid = np.genfromtxt(grid_path)
                    #grid = np.genfromtxt(grid_path)[:, [-1]]
                    #print(grid.shape)
                    label_list = np.genfromtxt(label_path, dtype=np.int32).tolist()
                    label = np.zeros(36)
                    label[label_list] = 1

                    if self.transform:
                        sample = {'grid': grid, 'label': label}
                        sample = self.transform(sample)

                    self.list_of_grid[index] = sample['grid']
                    self.list_of_label[index] = sample['label']

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_grid, self.list_of_label], f)
            else:
                print('loading data from processed data')
                with open(self.save_path, 'rb') as f:
                    self.list_of_grid, self.list_of_label = pickle.load(f)
        # else:

    def __len__(self):
        return len(self.grid_path)

    def __getitem__(self, index):
        return self.list_of_grid[index], self.list_of_label[index]


class VOXELDataset2(Dataset):
    def __init__(self, grid_path, label_path, transform=None):
        self.transform = transform
        self.grid = np.load(grid_path)
        self.label = np.load(label_path)

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, index):
        sample = {'grid': self.grid[index], 'label': self.label[index]}
        if self.transform:
            sample = self.transform(sample)

        return sample['grid'], sample['label']


class ToTensor(object):
    def __call__(self, sample):
        grid = sample['grid']
        label = sample['label']

        return {'grid': torch.from_numpy(grid).to(torch.float32),
                'label': torch.from_numpy(label).to(torch.float32)}


class To3DGrid(object):
    def __call__(self, sample):
        grid = sample['grid']
        label = sample['label']

        # grid = np.reshape(grid, (1, 40, 40, 40))        # ?????
        grid = np.reshape(grid, (1, 32, 32, 32))

        return {'grid': grid,
                'label': label}


def test():
    for root, _, files in os.walk('D:/Programfiles/SCVP/archive/SC_label_data/trainingdata'):
        if len(files) == 2:
            label_path = os.path.join(root, files[1])
            print(label_path)
            label = torch.zeros(64)
            label_list = np.genfromtxt(label_path, dtype=np.int64).tolist()
            label[label_list] = 1
            print(label)


def load_and_print_first_20_items(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            #for item in data[0]:  # 打印前20个数据项
            #    print(item)
            print(len(data[0]))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

#if __name__ == "__main__":
    '''file_path = 'D:/Programfiles/SCVP/archive/process_data.dat'  # 替换为 .dat 文件路径
    load_and_print_first_20_items(file_path)'''

if __name__ == "__main__":
    dataset = VOXELDataset('../data/Trainingdata', transform=transforms.Compose([To3DGrid(),ToTensor()]), processed_data=True)
    #dataset = VOXELDataset2('../grids.npy', '../labels.npy', transform=transforms.Compose([To3DGrid(), ToTensor()]))
    #test()




