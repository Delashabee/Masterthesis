import torch.nn as nn
import torch

from dataset import VOXELDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NBVLoss(nn.Module):
    def __init__(self, lambda_for0, lambda_for1):
        super(NBVLoss, self).__init__()

        self.mse = nn.MSELoss()
        self.entropy = nn.BCELoss()
        # self.l1loss = nn.L1Loss()
        # self.sigmoid = nn.Sigmoid()

        self.lambda_for0 = lambda_for0
        self.lambda_for1 = lambda_for1
        # self.lambda_l2   = 1
        # self.lambda_cnt1 = 1

    def forward(self, predictions, target):
        # Euclidean distance
        # l2loss = self.mse(predictions, target)

        # loss_where_1
        # index_1 = torch.nonzero(target)
        loss_where_1 = 0
        loss_where_0 = 0
        # for index in index_1:
        #     i, j = index[0].item(), index[1].item()
        #     loss_where_1 += self.entropy(predictions[i][j],  target[i][j])

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.entropy(predictions[i][j], target[i][j]).to(device)
                else:
                    loss_where_1 += self.entropy(predictions[i][j], target[i][j]).to(device)

        # cnt_target1 = torch.nonzero(target).shape[0]
        # cnt_pred1   = torch.nonzero(predictions[predictions > 0.5]).shape[0]

        # loss_cnt1 = torch.exp(torch.tensor(abs(cnt_target1-cnt_pred1)))
        # loss_cnt1 = torch.tensor(abs(cnt_target1-cnt_pred1)**2)

        return (
                self.lambda_for1 * loss_where_1
                + self.lambda_for0 * loss_where_0
            # + self.lambda_l2 * l2loss
            # + self.lambda_cnt1 * loss_cnt1
        )


class NBVLoss2(nn.Module):
    def __init__(self, lambda_for0, lambda_for1):
        super(NBVLoss2, self).__init__()

        self.mse = nn.MSELoss()
        self.entropy = nn.BCELoss()
        # self.l1loss = nn.L1Loss()
        # self.sigmoid = nn.Sigmoid()

        self.lambda_for0 = lambda_for0
        self.lambda_for1 = lambda_for1
        # self.lambda_l2   = 1
        # self.lambda_cnt1 = 1

    def forward(self, predictions, target):
        # Euclidean distance
        # l2loss = self.mse(predictions, target)

        # loss_where_1
        # index_1 = torch.nonzero(target)
        loss_where_1 = 0
        loss_where_0 = 0
        # for index in index_1:
        #     i, j = index[0].item(), index[1].item()
        #     loss_where_1 += self.entropy(predictions[i][j],  target[i][j])

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.mse(predictions[i][j], target[i][j]).to(device)
                else:
                    loss_where_1 += self.mse(predictions[i][j], target[i][j]).to(device)

        # cnt_target1 = torch.nonzero(target).shape[0]
        # cnt_pred1   = torch.nonzero(predictions[predictions > 0.5]).shape[0]

        # # loss_cnt1 = torch.exp(torch.tensor(abs(cnt_target1-cnt_pred1)))
        # loss_cnt1 = torch.tensor(abs(cnt_target1-cnt_pred1)**2)

        return (
                self.lambda_for1 * loss_where_1
                + self.lambda_for0 * loss_where_0
            # + self.lambda_l2 * l2loss
            # + self.lambda_cnt1 * loss_cnt1
        )


class NBVLoss3(nn.Module):
    def __init__(self, lambda_for0, lambda_for1):
        super(NBVLoss3, self).__init__()

        self.mse = nn.MSELoss()
        self.entropy = nn.BCEWithLogitsLoss()
        # self.l1loss = nn.L1Loss()
        # self.sigmoid = nn.Sigmoid()

        self.lambda_for0 = lambda_for0
        self.lambda_for1 = lambda_for1
        # self.lambda_l2   = 1
        # self.lambda_cnt1 = 1

    def forward(self, predictions, target):
        # Euclidean distance
        # l2loss = self.mse(predictions, target)

        # loss_where_1
        # index_1 = torch.nonzero(target)
        loss_where_1 = 0
        loss_where_0 = 0
        # for index in index_1:
        #     i, j = index[0].item(), index[1].item()
        #     loss_where_1 += self.entropy(predictions[i][j],  target[i][j])

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if target[i][j] == 0:
                    loss_where_0 += self.entropy(predictions[i][j], target[i][j]).to(device)
                else:
                    loss_where_1 += self.entropy(predictions[i][j], target[i][j]).to(device)

        # cnt_target1 = torch.nonzero(target).shape[0]
        # cnt_pred1   = torch.nonzero(predictions[predictions > 0.5]).shape[0]

        # loss_cnt1 = torch.exp(torch.tensor(abs(cnt_target1-cnt_pred1)))
        # loss_cnt1 = torch.tensor(abs(cnt_target1-cnt_pred1)**2)

        return (
                self.lambda_for1 * loss_where_1
                + self.lambda_for0 * loss_where_0
            # + self.lambda_l2 * l2loss
            # + self.lambda_cnt1 * loss_cnt1
        )


class MyLoss(nn.Module):
    def __init__(self, target):
        super(MyLoss, self).__init__()
        self.pos_weight = self.calculate_pos_weight(target)
        self.bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def calculate_pos_weight(self, target):
        total_elements = target.numel()
        num_positive = (target == 1).sum(dim=0).float()
        num_negative = (target == 0).sum(dim=0).float()
        pos_weight = num_negative / (num_positive + 1e-6)
        return pos_weight

    def forward(self, predictions, target):
        loss = self.bce_with_logits(predictions, target)
        return loss


class MyLoss2(nn.Module):
    def __init__(self, target):
        super(MyLoss2, self).__init__()
        self.pos_weight = self.calculate_class_weights(target)
        self.bce_with_logits = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def calculate_class_weights(self, target):
        # labels shape: (num_samples, num_classes)
        num_samples, num_classes = target.shape
        class_weights = []
        for i in range(num_classes):
            class_count = target[:, i].sum().item()
            weight = (num_samples - class_count) / (class_count + 1e-6)
            class_weights.append(weight)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights_tensor

    def forward(self, predictions, target):
        loss = self.bce_with_logits(predictions, target)
        return loss


class MyNBVLoss(nn.Module):
    def __init__(self, target):
        super(MyNBVLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.lambda_for0 = 1
        self.lambda_for1 = self.calculate_pos_weight(target)

    def calculate_pos_weight(self, target):
        total_elements = target.numel()
        num_positive = (target == 1).sum(dim=0).float()
        num_negative = (target == 0).sum(dim=0).float()
        pos_weight = num_negative / (num_positive + 1e-6)
        return pos_weight / 2

    def forward(self, predictions, target):
        loss = self.bce(predictions, target)

        loss_where_0 = loss * (target == 0).float()
        loss_where_1 = loss * (target == 1).float()

        weighted_loss_where_1 = loss_where_1 * self.lambda_for1

        total_loss = self.lambda_for0 * loss_where_0.sum() + weighted_loss_where_1.sum()

        return total_loss


if __name__ == "__main__":
    test_dataset = VOXELDataset('../data/novel_test_data2/Armadillo',
                                transform=transforms.Compose([To3DGrid(), ToTensor()]))
    loader = DataLoader(test_dataset, batch_size=64)
    dataiter = iter(loader)
    data1 = dataiter.next()

    grid, label = data1
    print(label)

    cnt1 = torch.nonzero(label)
    print(cnt1)
    print(cnt1.shape[0])

