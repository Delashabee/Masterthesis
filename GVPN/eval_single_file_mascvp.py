import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from dataset import VOXELDataset, VOXELDataset2, ToTensor, To3DGrid
from model import MyNBVNetV1, MyNBVNetV2, MyNBVNetV3, MyNBVNetV5
from torch.autograd import Variable
import sys
import time
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dictionary to store the features
outputs = {}


# Hook function to capture the features
def hook_fn(module, input, output):
    outputs['features'] = output


# Visualize feature maps
def visualize_feature_maps(features):
    features = features.cpu().detach().numpy()[0]  # Take the first element of the batch
    num_channels, depth, height, width = features.shape

    # Create a grid of plots
    fig, axes = plt.subplots(nrows=depth, ncols=num_channels, figsize=(num_channels * 2, depth * 2))
    fig.suptitle("Features Before FC Layer")

    for d in range(depth):
        for c in range(num_channels):
            # Normalize the feature map to 0-255
            feature_map = features[c, d]
            normalized_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-6)
            gray_map = (normalized_map * 255).astype(np.uint8)

            if num_channels > 1:
                axes[d, c].imshow(gray_map, cmap='gray')
                axes[d, c].axis('off')
            else:
                axes[d].imshow(gray_map, cmap='gray')
                axes[d].axis('off')

    plt.tight_layout()
    plt.show()


def eval(datapath, label_path, model_path):
    test_data = np.genfromtxt(datapath).reshape(1, 1, 32, 32, 32)
    test_data = torch.from_numpy(test_data).to(torch.float32)
    label_list = np.genfromtxt(label_path, dtype=np.int32).tolist()
    label = np.zeros(36)
    label[label_list] = 1

    model = MyNBVNetV1(num_classes=36)
    model = model.to(device)



    checkpoint = torch.load(model_path,
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    print('EVALUATING')
    model.eval()
    grid = test_data.to(device)

    startTime = time.time()

    output = model(grid)

    endTime = time.time()
    print('run time is ' + str(endTime - startTime))
    # np.savetxt('./run_time/'+name_of_model+'.txt',np.asarray([endTime-startTime]))
    print('output:', output[0])
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    correct1 = 0
    wrong1 = 0
    cnt1 = 0
    recall = 0
    precision = 0
    for j in range(36):
        if label[j] == 1 and output[0][j] == 1:
            correct1 += 1
            cnt1 += 1
        elif label[j] == 1 and output[0][j] == 0:
            cnt1 += 1
        elif label[j] == 0 and output[0][j] == 1:
            wrong1 += 1
    recall += (correct1 / cnt1)
    precision += (correct1 / (correct1 + wrong1 + 1e-6))
    print(output.shape)
    print('recall:', recall, 'precision:', precision)


    return output


if __name__ == '__main__':
    name_of_model = '44'
    #label_path = 'Boneviewsid000.txt'
    model_path = 'D:/Programfiles/Myscvp/SCVPNet/traindata36_NBVNet1_Adam_NBVLoss_la0_1_la1_2.25_e150_b32_l0.0002_gamma0.5.pth.tar'
    pred = eval('D:/Programfiles/Myscvp/industrial_label_data/36_views/36_views/Trainingdata/044/toward0_rotate0_view0/grid_toward0_rotate0_view0.txt',
                'D:/Programfiles/Myscvp/industrial_label_data/36_views/36_views/Trainingdata/044/toward0_rotate0_view0/ids_toward0_rotate0_view0.txt',
                model_path)
    ans = []
    for i in range(pred.shape[1]):
        if pred[0][i] == 1:
            print(i)
            ans.append(i)
    print('ans:', ans)
    np.savetxt('./log/' + name_of_model + '.txt', ans, fmt='%d')
