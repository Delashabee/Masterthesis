# Name: Jiaming Yu
# Time:
import torch
from torch.utils.data import Dataset, DataLoader
from model import MyNBVNetV1, MyNBVNetV2, MyNBVNetV3, MyNBVNetV4, MyNBVNetV5, MyNBVNetV6, VoxResNet, \
    TransformerClassifier, MyNBVNetV7, ResNeXt
import argparse
import pickle
import numpy as np

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
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in testing[default: 32]')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma in testing[default: 0.5]')
    parser.add_argument('--model', default='NBVNet1', help='model name [default: NBVNet1]')
    parser.add_argument('--model_path', type=str, default='./Final_SCVP64_my_checkpoint_forevery10epochs.pth.tar',
                        help='path to stored model parameters[default: ./model.pth.tar]')
    parser.add_argument('--num_classes', type=int, default=40, help='number of classes in testing[default: 36]')
    parser.add_argument('--test_data_path', type=str, default='test_data.dat', help='path to test data file')

    return parser.parse_args()


def check_accuracy(model, test_loader, gamma, num_classes):
    print('EVALUATING ON TEST SET')
    model.eval()
    accuracy_exp = 0
    recall = 0
    precision = 0
    accuracy = []
    test_case_num = len(test_loader.dataset)

    with torch.no_grad():
        for sample in test_loader:
            grid = sample[0].to(device)
            label = sample[1].to(device)
            output = model(grid)
            output[output >= gamma] = 1
            output[output < gamma] = 0
            for i in range(label.shape[0]):
                correct1 = 0
                wrong1 = 0
                cnt1 = 0
                for j in range(num_classes):
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
    print(f'recall: {recall:.4f}, precision: {precision:.4f}, F1_score: {F1_score:.4f}, accuracy: {mean_acc:.4f}')

    return recall, precision, F1_score, mean_acc


def load_checkpoint(path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'Loaded checkpoint from epoch: {checkpoint["epoch"]}')


def main(opt):
    print('------LOADING TEST DATA------')
    dataset = ProcessedDataset(opt.test_data_path)
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

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

    load_checkpoint(opt.model_path, model)
    recall, precision, F1_score, accuracy = check_accuracy(model, test_loader, opt.gamma, opt.num_classes)
    print(f'Final Results - Recall: {recall:.4f}, Precision: {precision:.4f}, F1 Score: {F1_score:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
