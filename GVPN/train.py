import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from model import MyNBVNetV1, MyNBVNetV2, MyNBVNetV3, MyNBVNetV4, MyNBVNetV5, MyNBVNetV6, VoxResNet, \
    TransformerClassifier, MyNBVNetV7, ResNeXt
from torch.autograd import Variable
from loss import NBVLoss, NBVLoss2, NBVLoss3, MyLoss, MyNBVLoss, MyLoss2
import argparse
import os
import pickle
import wandb
import torch.optim.lr_scheduler as lr_scheduler

shuffle_dataset = True
random_seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class ProcessedDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as file:
            self.grid_data, self.label_data = pickle.load(file)

    def __len__(self):
        return len(self.grid_data)

    def __getitem__(self, idx):
        return self.grid_data[idx], self.label_data[idx]


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training[default: 32]')
    parser.add_argument('--lambda_0', type=float, default=1, help='lambda_0 in training[default: 1]')
    parser.add_argument('--lambda_1', type=float, default=1, help='lambda_1 in training[default: 1]')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma in training[default: 0.5]')
    parser.add_argument('--model', default='NBVNet1', help='model name [default: NBVNet1]')
    parser.add_argument('--epochs', default=150, type=int, help='number of epoch in training [default: 150]')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training[default: Adam]')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--validation_split', default=0.2, type=float,
                        help='split rate for validation data[default: 0.2]')
    parser.add_argument('--loss', type=str, default='NBVLoss', help='specify the loss for the model[default: NBVLoss]')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='load stored model parameters for training[default: False]')

    parser.add_argument('--model_path', type=str, default='./Final_SCVP64_my_checkpoint_forevery10epochs.pth.tar',
                        help='path to stored model parameters[default: ./model.pth.tar]')
    parser.add_argument('--num_classes', type=int, default=36, help='nummber of classes in training[default: 36]')

    return parser.parse_args()


def train_fn(loader, model, optimizer, criterion):
    print('------Training------')
    loop = tqdm(loader, leave=True)
    losses = []

    for batch_idx, data in enumerate(loop):
        grid = data[0]
        grid = grid.to(device)
        label = data[1]
        label = label.to(device)

        output = model(grid)
        loss = criterion(output, label)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    mean_loss = sum(losses) / len(losses)
    return mean_loss


def check_accuracy(model, test_loader, test_case_num):
    print('EVALUATING')
    model.eval()
    accuracy_exp = 0
    recall = 0
    precision = 0
    accuracy = []
    for sample in test_loader:
        grid = sample[0].to(device)
        label = sample[1].to(device)
        output = model(grid)
        output[output >= opt.gamma] = 1
        output[output < opt.gamma] = 0
        for i in range(label.shape[0]):
            correct1 = 0
            wrong1 = 0
            cnt1 = 0
            for j in range(opt.num_classes):
                if label[i][j] == 1 and output[i][j] == 1:
                    correct1 += 1
                    cnt1 += 1
                elif label[i][j] == 1 and output[i][j] == 0:
                    cnt1 += 1
                elif label[i][j] == 0 and output[i][j] == 1:
                    wrong1 += 1

            correct_exp = (output[i] == label[i]).sum().item()
            accuracy_exp += 1 / np.exp(64 - correct_exp)
            recall += (correct1 / cnt1)
            precision += (correct1 / (correct1 + wrong1 + 1e-6))

        correct = (output == label).sum().item()
        acc = correct / (output.shape[0] * output.shape[1])
        accuracy.append(acc)

    accuracy_exp /= test_case_num
    recall /= test_case_num
    precision /= test_case_num
    F1_score = 2 * recall * precision / (recall + precision + 1e-6)
    mean_acc = sum(accuracy) / len(accuracy)
    print(f'recall:{recall}, precision:{precision}, F1_score:{F1_score}, accuracy:{mean_acc}')
    model.train()
    return recall, precision, F1_score, mean_acc


def load_checkpoint(path, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print(f'loaded epoch: {checkpoint["epoch"]}')
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        'epoch': epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def main(opt, processed_data_path):
    api_key = '83cdff602bf85f8046129aef062fcdb8741c4381'
    wandb.login(key=api_key)
    wandb.init(project="SCVP64", entity="master_",
               name=f"traindata{opt.num_classes}_{opt.model}_{opt.optimizer}_{opt.loss}_la0_{opt.lambda_0}_la1_{opt.lambda_1}_e{opt.epochs}_b{opt.batch_size}_l{opt.learning_rate}_gamma{opt.gamma}classes{opt.num_classes}")

    print('------LOADING DATA------')
    dataset = ProcessedDataset(processed_data_path)

    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    num_epochs = opt.epochs
    validation_split = opt.validation_split

    dataset_size = len(dataset)
    print(f'len(dataset) : {dataset_size}')

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    model_mapping = {
        'NBVNet1': lambda: MyNBVNetV1(num_classes=opt.num_classes),
        'NBVNet2': MyNBVNetV2,
        'NBVNet3': lambda: MyNBVNetV3(num_classes=opt.num_classes),
        'NBVNet4': MyNBVNetV4,
        'NBVNet5': MyNBVNetV5,
        'NBVNet6': MyNBVNetV6,
        'NBVNet7': MyNBVNetV7,
        'VoxResNet': VoxResNet,
        'Transformer': lambda: TransformerClassifier(feature_dim=128, nhead=4, num_encoder_layers=6, num_classes=64),
        'ResNeXt': lambda: ResNeXt(layers=[3, 4, 6, 3], cardinality=32, num_classes=opt.num_classes),
        'ResNeXt_101': lambda: ResNeXt(layers=[3, 4, 23, 3], cardinality=32, num_classes=opt.num_classes)
    }

    if opt.model not in model_mapping:
        raise ValueError(f"Unknown model: {opt.model}")

    print(f'Model : {opt.model}')
    model = model_mapping[opt.model]().to(device)
    wandb.watch(model)

    optimizer_mapping = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
    }

    if opt.optimizer not in optimizer_mapping:
        raise ValueError(f"Unknown optimizer: {opt.optimizer}")

    print(f'Optimizer: {opt.optimizer}')
    optimizer = optimizer_mapping[opt.optimizer](model.parameters(), lr=learning_rate)

    loss_mapping = {
        'MyLoss': lambda: MyLoss(torch.cat([dataset[i][1].unsqueeze(0) for i in train_indices], dim=0).to(device)),
        'MyLoss2': lambda: MyLoss2(torch.cat([dataset[i][1].unsqueeze(0) for i in train_indices], dim=0).to(device)),
        'MyNBVLoss': lambda: MyNBVLoss(
            torch.cat([dataset[i][1].unsqueeze(0) for i in train_indices], dim=0).to(device)),
        'NBVLoss': lambda: NBVLoss(opt.lambda_0, opt.lambda_1),
        'MSELoss': nn.MSELoss,
        'BCELoss': nn.BCELoss,
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
        'NBVLoss2': lambda: NBVLoss2(opt.lambda_0, opt.lambda_1),
        'NBVLoss3': lambda: NBVLoss3(opt.lambda_0, opt.lambda_1)
    }

    if opt.loss not in loss_mapping:
        raise ValueError(f"Unknown loss: {opt.loss}")

    print(f'Loss: {opt.loss}')
    criterion = loss_mapping[opt.loss]().to(device)

    if opt.load_model:
        print('Loading saved model')
        load_checkpoint(opt.model_path, model, optimizer, learning_rate)
    else:
        print('Training new model')

    best_F1_score = 0.0
    checkpoint_filename = f"traindata{opt.num_classes}_{opt.model}_{opt.optimizer}_{opt.loss}_la0_{opt.lambda_0}_la1_{opt.lambda_1}_e{opt.epochs}_b{opt.batch_size}_l{opt.learning_rate}_gamma{opt.gamma}classes{opt.num_classes}.pth.tar"

    for epoch in range(num_epochs):
        print(f'epoch : {epoch + 1}/{num_epochs}')
        mean_loss = train_fn(train_loader, model, optimizer, criterion)

        '''if (epoch + 1) % 10 == 0:
            checkpoint_filename = f"traindata_{opt.model}_{opt.optimizer}_{opt.loss}_la0_{opt.lambda_0}_la1_{opt.lambda_1}_e{opt.epochs}_b{opt.batch_size}_l{opt.learning_rate}_gamma{opt.gamma}_epoch{epoch+1}.pth.tar"
            save_checkpoint(model, optimizer, epoch, filename=checkpoint_filename)'''

        print('On test loader')
        recall, precision, F1_score, test_accuracy = check_accuracy(model, validation_loader, len(val_indices))
        wandb.log({"recall": recall, "precision": precision, "F1_score": F1_score, "test_accuracy": test_accuracy,
                   "mean_loss": mean_loss})

        if F1_score > best_F1_score:
            best_F1_score = F1_score
            save_checkpoint(model, optimizer, epoch, filename=checkpoint_filename)
            print(f'Saved new best model with F1 score: {best_F1_score}')


if __name__ == "__main__":
    opt = parse_args()
    processed_data_path = f"/home/jovyan/Master_thesis/src/data/train{opt.num_classes}.dat"
    main(opt, processed_data_path)
